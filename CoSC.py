import torch
import util
from args import TestArgParser
from saver import ModelSaver
from openpyxl import load_workbook
import time
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _kernel_weights(query_summary, cal_summary, h, kernel):
    query_summary = _as_summary_matrix(query_summary)
    cal_summary = _as_summary_matrix(cal_summary)
    dist = _normalized_distance(query_summary, cal_summary)

    return _kernel_weights_from_distance(dist, h, kernel)

def _total_variance_scale(cal_summary):
    """Return sqrt(sum of calibration-feature variances)."""
    cal_summary = _as_summary_matrix(cal_summary)
    total_variance = cal_summary.var(dim=0, correction=0).sum()
    eps = torch.finfo(cal_summary.dtype).eps

    return total_variance.clamp_min(eps).sqrt()

def _normalized_distance(query_summary, cal_summary):
    """Scale distances using only the calibration sample statistics."""
    query_summary = _as_summary_matrix(query_summary)
    cal_summary = _as_summary_matrix(cal_summary)
    scale = _total_variance_scale(cal_summary)

    # Direct differences preserve zero self-distances and keep weights stable
    # across query batch sizes when the same H updates calibration and test.
    return torch.cdist(query_summary, cal_summary,
                       compute_mode='donot_use_mm_for_euclid_dist') / scale

def _kernel_weights_from_distance(dist, h, kernel):

    if kernel == 'gaussian':
        return torch.clamp(torch.exp(-(dist ** 2) / (2 * h ** 2)), 0, 1)

    if kernel == 'laplacian':
        return torch.clamp(torch.exp(-dist / h), 0, 1)

    if kernel == 'epanechikov':
        return torch.clamp(1 - (dist/h)**2, min=0)

    if kernel == 'triangular':
        return torch.clamp(1 - dist/h, min=0)
    
    raise ValueError(f"Unsupported kernel: {kernel}")

def _as_summary_matrix(summary):
    if summary.ndim == 1:
        return summary.reshape(-1, 1)
    
    if summary.ndim == 2:
        return summary

    raise ValueError("summary must be 1-D or 2-D")

def _predicted_median(cdf_matrix, time_grid, tau=None, level=0.5):
    if cdf_matrix.ndim != 2 or time_grid.ndim != 1:
        raise ValueError("cdf_matrix must be 2-D and time_grid must be 1-D")
    
    if cdf_matrix.shape[1] != time_grid.numel():
        raise ValueError("cdf_matrix columns must match time_grid")

    if torch.any(time_grid[1:] < time_grid[:-1]):
        raise ValueError("time_grid must be sorted")

    if tau is None:
        tau = time_grid[-1]
    tau = torch.as_tensor(tau, dtype=time_grid.dtype, device=time_grid.device)

    crosses = cdf_matrix >= level
    has_cross = crosses.any(dim=1)
    right = crosses.float().argmax(dim=1)
    left = (right - 1).clamp_min(0)

    right_t = time_grid[right]
    left_t = torch.where(right > 0, time_grid[left], time_grid.new_zeros(right.shape))
    right_cdf = cdf_matrix.gather(1, right.unsqueeze(1)).squeeze(1)
    left_cdf = torch.where(
        right > 0,
        cdf_matrix.gather(1, left.unsqueeze(1)).squeeze(1),
        cdf_matrix.new_zeros(right.shape),
    )

    denom = (right_cdf - left_cdf).clamp_min(torch.finfo(cdf_matrix.dtype).eps)
    fraction = ((level - left_cdf) / denom).clamp(0, 1)
    median = left_t + fraction * (right_t - left_t)
    median = torch.where(has_cross, median, tau.expand_as(median))
    median = median.clamp(min=0, max=tau.item())

    return median / tau.clamp_min(torch.finfo(time_grid.dtype).eps)

def _fit_pca_components(args, fit_src, eps=1e-8):
    fit_src = fit_src.float()
    _, s, vh = torch.linalg.svd(fit_src, full_matrices=False)
    explained_variance = s.pow(2)
    explained_ratio = explained_variance / explained_variance.sum().clamp_min(eps)

    pca_var = getattr(args, 'pit_cond_pca_var', 0.0)
    max_components = min(fit_src.shape[0], fit_src.shape[1])
    if pca_var > 0:
        if pca_var > 1:
            raise ValueError("pit_cond_pca_var must be in (0, 1]")
        n_components = int(torch.searchsorted(torch.cumsum(explained_ratio, dim=0), pca_var).item()) + 1

    else:
        n_components = args.pit_cond_pca_dim
    n_components = min(n_components, max_components)

    if n_components <= 0:
        raise ValueError("pit_cond_pca_dim must be positive when pit_cond_pca_var is 0")

    components = vh[:n_components].T

    return components, explained_ratio[:n_components]

def _project_pca(args, fit_src, query_src, verbose=False):
    fit = fit_src.float()
    query = query_src.float()

    components, selected_ratio = _fit_pca_components(args, fit)
    if verbose:
        print("PCA selected components:", components.shape[1])
        print(
            "PCA explained variance ratio:",
            [round(x, 6) for x in selected_ratio.detach().cpu().tolist()],
        )
        print("PCA cumulative explained variance:", round(selected_ratio.sum().detach().cpu().item(), 6))

    return fit @ components, query @ components

def _pca_summaries(args, src_valid, src_test):
    valid_summary, test_summary = _project_pca(args, src_valid, src_test, verbose=True)

    return valid_summary, test_summary

def _conditional_summaries(args, cdf_valid, time_valid, src_valid, cdf_test, time_test, src_test, tau):
    summary = getattr(args, 'pit_cond_summary', 'rmst')
    if summary == 'rmst':
        return util.normalized_rmst(cdf_valid, time_valid, tau=tau), util.normalized_rmst(cdf_test, time_test, tau=tau)

    if summary == 'median':
        return _predicted_median(cdf_valid, time_valid, tau=tau), _predicted_median(cdf_test, time_test, tau=tau)

    if summary == 'pca':
        return _pca_summaries(args, src_valid, src_test)

    if summary == 'raw':
        return src_valid.float(), src_test.float()

    raise ValueError(f"Unsupported pit_cond_summary: {summary}")

def _stratified_folds(is_dead, n_folds, seed):
    device = is_dead.device
    event_idx = torch.nonzero(is_dead.reshape(-1) == 1, as_tuple=False).reshape(-1)
    cens_idx = torch.nonzero(is_dead.reshape(-1) == 0, as_tuple=False).reshape(-1)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    def shuffled(indices):
        if indices.numel() == 0:
            return indices

        order = torch.randperm(indices.numel(), generator=generator).to(device)

        return indices[order]

    event_folds = torch.tensor_split(shuffled(event_idx), n_folds)
    cens_folds = torch.tensor_split(shuffled(cens_idx), n_folds)
    folds = []
    for event_fold, cens_fold in zip(event_folds, cens_folds):
        fold = torch.cat([event_fold, cens_fold])
        if fold.numel() > 0:
            order = torch.randperm(fold.numel(), generator=generator).to(device)
            fold = fold[order]
        folds.append(fold)

    return folds

def _prepare_weighted_ecdf(cdf, is_dead, query, eps=1e-8):
    """Cache each calibration sample's ECDF contribution per query."""
    observed = cdf.reshape(-1)
    event = is_dead.reshape(-1).float()
    query = query.reshape(-1).contiguous()
    inverse_survival = (1 - event) / (1 - observed).clamp_min(eps)
    eligible = observed.unsqueeze(0) <= query.unsqueeze(1)
    contribution = eligible * (
        event.unsqueeze(0)
        + inverse_survival.unsqueeze(0) * (query.unsqueeze(1) - observed.unsqueeze(0))
    )

    return {
        'contribution': contribution,
        'eps': eps,
    }

def _weighted_ecdf_from_prepared(prepared, weights):
    """Evaluate the weighted ECDF using cached pointwise contributions."""
    transformed = (weights * prepared['contribution']).sum(dim=1)
    transformed = transformed / weights.sum(dim=1).clamp_min(prepared['eps'])

    return transformed.clamp(0, 1)

def _prepare_cv_fold_contexts(args, cdf_valid, is_dead_valid, weight_features_valid, folds, src_valid=None):
    """Cache fold-specific projections and distances shared by every bandwidth."""
    n = cdf_valid.shape[0]
    use_fold_pca = getattr(args, 'pit_cond_summary', 'rmst') == 'pca' and src_valid is not None
    contexts = []
    for fold in folds:
        tune_mask = torch.zeros(n, dtype=torch.bool, device=cdf_valid.device)
        tune_mask[fold] = True
        cal_mask = ~tune_mask

        if use_fold_pca:
            features_cal, features_tune = _project_pca(args, src_valid[cal_mask], src_valid[tune_mask], verbose=False)

        else:
            features_cal = weight_features_valid[cal_mask]
            features_tune = weight_features_valid[tune_mask]

        tune_idx = torch.nonzero(tune_mask, as_tuple=False).reshape(-1)
        cdf_tune = cdf_valid[tune_mask]
        cdf_cal = cdf_valid[cal_mask]
        is_dead_cal = is_dead_valid[cal_mask]
        contexts.append({
            'tune_idx': tune_idx,
            'cdf_tune': cdf_tune,
            'is_dead_tune': is_dead_valid[tune_mask],
            'cdf_cal': cdf_cal,
            'is_dead_cal': is_dead_cal,
            'ecdf': _prepare_weighted_ecdf(cdf_cal, is_dead_cal, cdf_tune),
            # Normalize with calibration-fold statistics only, avoiding
            # held-out-fold leakage while making h dimensionless.
            'distance': _normalized_distance(features_tune, features_cal),
        })

    return contexts

def _cv_post_calibration_loss(args, cdf_valid, is_dead_valid, weight_features_valid, folds, h, src_valid=None, fold_contexts=None):
    if fold_contexts is None:
        fold_contexts = _prepare_cv_fold_contexts(
            args,
            cdf_valid,
            is_dead_valid,
            weight_features_valid,
            folds,
            src_valid=src_valid,
        )

    weighted_loss = h.new_tensor(0.0) if torch.is_tensor(h) else torch.tensor(0.0, device=cdf_valid.device)
    weighted_count = 0
    for context in fold_contexts:
        cal_weights = _kernel_weights_from_distance(context['distance'], h, args.kernel)
        transformed_fold = _weighted_ecdf_from_prepared(context['ecdf'], cal_weights)
        # Match the KS implementation used by the reported test metric while
        # evaluating each CV fold separately.
        _, fold_loss = util.get_p_value(args=args, cdf=transformed_fold, is_dead=context['is_dead_tune'], device=cdf_valid.device)
        fold_size = int(context['tune_idx'].numel())
        weighted_loss = weighted_loss + fold_loss * fold_size
        weighted_count += fold_size

    return weighted_loss / weighted_count

def _fit_bandwidth(args, cdf_valid, is_dead_valid, weight_features_valid, src_valid=None):
    """Grid-search h, with optional caller-supplied candidates, then refine."""
    n = cdf_valid.shape[0]
    weight_features_valid = _as_summary_matrix(weight_features_valid)
    feature_dim = weight_features_valid.shape[1]

    n_folds = min(args.pit_cond_cv_folds, n)
    cv_repeats = getattr(args, 'pit_cond_cv_repeats', 1)

    if n_folds < 2:
        raise ValueError(
            "pit_cond_cv_folds must be at least 2 and validation set "
            "must have at least 2 samples"
        )
    if cv_repeats < 1:
        raise ValueError("pit_cond_cv_repeats must be positive")

    refinement_iters = args.pit_cond_max_iters
    if refinement_iters < 0:
        raise ValueError("pit_cond_max_iters must be non-negative")

    # Every candidate is evaluated on exactly the same repeated partitions.
    fold_sets = [_stratified_folds(is_dead_valid, n_folds, args.seed + repeat_idx) for repeat_idx in range(cv_repeats)]
    supplied_candidates = getattr(args, 'pit_cond_bandwidth_candidates', None)
    if supplied_candidates is None:
        candidates = (
            torch.arange(10, 51, device=cdf_valid.device, dtype=cdf_valid.dtype)
            / 100
        )
        grid_description = "range=[0.1, 0.5], step=0.01"
    else:
        candidates = torch.as_tensor(
            supplied_candidates, device=cdf_valid.device, dtype=cdf_valid.dtype
        ).reshape(-1)
        if (candidates.numel() == 0 or torch.isnan(candidates).any()
                or torch.any(candidates <= 0)
                or torch.any(candidates[1:] <= candidates[:-1])):
            raise ValueError(
                "pit_cond_bandwidth_candidates must be strictly increasing "
                "positive values; +inf is allowed"
            )
        grid_description = (
            f"custom range=[{float(candidates[0]):g}, "
            f"{float(candidates[-1]):g}]"
        )

    coarse_steps = candidates.numel()

    loss_label = f"repeated CV({n_folds}-fold, repeats={cv_repeats}) post-PIT KS"
    print(
        "Grid-searching bandwidth",
        f"(loss={loss_label}, feature_dim={feature_dim}, "
        f"{grid_description}, coarse_steps={coarse_steps})"
    )

    global_best_h = float(candidates[0].item())
    global_best_loss = float('inf')
    evaluated = {}

    def evaluate(candidate_values, stage):
        nonlocal global_best_h, global_best_loss
        active_candidates = []
        active_indices = []
        for candidate_idx, h in enumerate(candidate_values):
            h_value = float(h.item())
            key = round(h_value, 12)
            if key in evaluated:
                continue

            active_candidates.append(h)
            active_indices.append(candidate_idx)

        if not active_candidates:
            return

        loss_sums = [cdf_valid.new_tensor(0.0) for _ in active_candidates]
        # Process one repeat at a time so cached O(n^2) distance matrices do
        # not accumulate across repeats. Every candidate within the repeat
        # reuses exactly the same PCA projections and pairwise distances.
        for folds in fold_sets:
            fold_contexts = _prepare_cv_fold_contexts(
                args,
                cdf_valid,
                is_dead_valid,
                weight_features_valid,
                folds,
                src_valid=src_valid,
            )
            for active_idx, h in enumerate(active_candidates):
                repeat_loss = _cv_post_calibration_loss(
                    args,
                    cdf_valid,
                    is_dead_valid,
                    weight_features_valid,
                    folds,
                    h,
                    src_valid=src_valid,
                    fold_contexts=fold_contexts,
                )
                loss_sums[active_idx] = loss_sums[active_idx] + repeat_loss

            del fold_contexts

        for h, candidate_idx, loss_sum in zip(active_candidates, active_indices, loss_sums):
            h_value = float(h.item())
            key = round(h_value, 12)
            loss = loss_sum / len(fold_sets)
            loss_value = float(loss.item())
            evaluated[key] = {'h': h_value, 'mean_loss': loss_value}
            print(
                f"h={h_value:g}, {loss_label} loss={loss_value:.6f}, "
                f"{stage}={candidate_idx + 1}/{candidate_values.numel()}", end='\r'
            )
            if math.isfinite(loss_value) and loss_value < global_best_loss:
                global_best_h = h_value
                global_best_loss = loss_value

    with torch.no_grad():
        evaluate(candidates, "coarse")

    if refinement_iters > 0:
        finite_candidates = candidates[torch.isfinite(candidates)]
        finite_results = [
            evaluated[round(float(h.item()), 12)] for h in finite_candidates
        ]
        finite_result_indices = [
            index for index, result in enumerate(finite_results)
            if math.isfinite(result['mean_loss'])
        ]
        coarse_best_finite = (
            min((finite_results[index] for index in finite_result_indices),
                key=lambda result: result['mean_loss'])
            if finite_result_indices else None
        )

        def refine_from(start_h, bounds=None):
            nonlocal global_best_h, global_best_loss

            if bounds is None:
                parameter = torch.nn.Parameter(
                    candidates.new_tensor(math.log(start_h)))

                def current_h():
                    return torch.exp(parameter)

                refinement_label = 'unbounded log-h'
            else:
                left, right = bounds
                position = (start_h - left) / (right - left)
                # Keep a nonzero derivative when the grid optimum is an endpoint.
                position = min(max(position, 0.01), 0.99)
                theta = math.log(position / (1 - position))
                parameter = torch.nn.Parameter(candidates.new_tensor(theta))

                def current_h():
                    return left + (right - left) * torch.sigmoid(parameter)

                refinement_label = f'bounded h in [{left:g}, {right:g}]'

            print(
                "\nGradient-refining bandwidth",
                f"(start=finite-grid-best, {refinement_label}, "
                f"h_init={start_h:g}, iterations={refinement_iters})", end='\r'
            )

            optimizer = torch.optim.Adam([parameter], lr=args.pit_cond_lr)
            refinement_best_loss = float('inf')
            patience_counter = 0

            for iter_idx in range(refinement_iters):
                # PIT_cond is invoked under torch.no_grad() in main. Re-enable
                # autograd only for the one-dimensional bandwidth refinement.
                with torch.enable_grad():
                    optimizer.zero_grad()
                    repeat_losses = []
                    h_value = float(current_h().detach().item())

                    # Build and release one repeat at a time. Keeping all
                    # repeated fold-distance matrices on the GPU causes OOM.
                    for folds in fold_sets:
                        with torch.no_grad():
                            fold_contexts = _prepare_cv_fold_contexts(
                                args,
                                cdf_valid,
                                is_dead_valid,
                                weight_features_valid,
                                folds,
                                src_valid=src_valid,
                            )
                        h_repeat = current_h()
                        repeat_loss = _cv_post_calibration_loss(
                            args,
                            cdf_valid,
                            is_dead_valid,
                            weight_features_valid,
                            folds,
                            h_repeat,
                            src_valid=src_valid,
                            fold_contexts=fold_contexts,
                        )
                        (repeat_loss / len(fold_sets)).backward()
                        repeat_losses.append(repeat_loss.detach())
                        del fold_contexts, repeat_loss

                    loss = torch.stack(repeat_losses).mean()
                    loss_value = float(loss.item())
                    if (parameter.grad is None
                            or not torch.isfinite(parameter.grad).all()):
                        print(
                            "\nNon-finite bandwidth gradient; "
                            "stopping refinement."
                        )
                        break

                    optimizer.step()

                print(
                    f"start=finite-grid-best, h={h_value:g}, "
                    f"{loss_label} loss={loss_value:.6f}, "
                    f"iteration={iter_idx + 1}/{refinement_iters}",
                    end='\r',
                )

                if (math.isfinite(loss_value)
                        and loss_value < refinement_best_loss):
                    refinement_best_loss = loss_value
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Retain every visited point. The final comparison also keeps
                # every coarse-grid candidate, including exact global weights.
                if math.isfinite(loss_value):
                    key = round(h_value, 12)
                    evaluated[key] = {'h': h_value, 'mean_loss': loss_value}
                    if loss_value < global_best_loss:
                        global_best_h = h_value
                        global_best_loss = loss_value

                if patience_counter >= args.pit_cond_patience:
                    break

        if (getattr(args, 'pit_cond_bounded_refinement', False)
                and coarse_best_finite is not None):
            best_index = min(
                finite_result_indices,
                key=lambda index: finite_results[index]['mean_loss'])
            left_index = max(0, best_index - 1)
            right_index = min(len(finite_candidates) - 1, best_index + 1)
            left = float(finite_candidates[left_index].item())
            right = float(finite_candidates[right_index].item())
            if right > left:
                refine_from(coarse_best_finite['h'], bounds=(left, right))
        elif (not getattr(args, 'pit_cond_bounded_refinement', False)
              and math.isfinite(global_best_h)):
            refine_from(global_best_h)

    if not math.isfinite(global_best_loss):
        raise RuntimeError("No bandwidth candidate produced a finite CV loss")

    best = min(
        (result for result in evaluated.values()
         if math.isfinite(result['mean_loss'])),
        key=lambda result: result['mean_loss'])
    best_h = best['h']
    best_loss = best['mean_loss']

    print(
        "Selected grid-search/gradient-refined optimum "
        f"h={best_h:g} with {loss_label} loss={best_loss:.6f} "
        f"(feature_dim={feature_dim})"
    )
    return best_h, best_loss


def PIT_cond(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    train_loader = util.get_train_loader(args, during_training=False)
    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    valid_loader, test_loader = eval_loaders

    # args.loss_fn = 'mle'

    for _, tgt_train in train_loader:
        tte_train = tgt_train[:, 0]
        is_dead_train = tgt_train[:, 1]

    for src_valid, tgt_valid in valid_loader:
        src_valid = src_valid.to(DEVICE)
        tgt_valid = tgt_valid.to(DEVICE)
        tte_valid = tgt_valid[:, 0]
        is_dead_valid = tgt_valid[:, 1]
        order_valid = torch.argsort(tte_valid)

    for src_test, tgt_test in test_loader:
        src_test = src_test.to(DEVICE)
        tgt_test = tgt_test.to(DEVICE)
        tte_test = tgt_test[:, 0]
        is_dead_test = tgt_test[:, 1]
        order_test = torch.argsort(tte_test)

    pred_params_valid = model.forward(src_valid)
    pred_params_test = model.forward(src_test)
    if args.model_dist in ['cat', 'mtlr', 'psr']:
        tgt_valid = util.cat_bin_target(args, tgt_valid, args.bin_boundaries)
        tgt_test = util.cat_bin_target(args, tgt_test, args.bin_boundaries)
        
    cdf_valid_all = util.get_cdf_matrix(pred_params_valid, tgt_valid, args)
    cdf_test_all = util.get_cdf_matrix(pred_params_test, tgt_test, args)

    cdf_valid = util.get_cdf_val(pred_params_valid, tgt_valid, args)

    if args.model_dist != 'cox':
        cdf_valid_all = cdf_valid_all[:, order_valid][order_valid, :]
        cdf_test_all = cdf_test_all[:, order_test][order_test, :]
        cdf_valid = cdf_valid[order_valid]

    else:
        cdf_valid_all = cdf_valid_all
        cdf_test_all = cdf_test_all
        cdf_valid = cdf_valid
        
    tte_valid = tte_valid[order_valid]
    is_dead_valid = is_dead_valid[order_valid]
    src_valid = src_valid[order_valid]

    tte_test = tte_test[order_test]
    is_dead_test = is_dead_test[order_test]
    src_test = src_test[order_test]

    start_time = time.time()
    tau = max(tte_train.max().item(), tte_valid.max().item())
    summary_valid, summary_test = _conditional_summaries(args, cdf_valid_all, tte_valid, src_valid, cdf_test_all, tte_test, src_test, tau=tau)

    best_h, best_loss = _fit_bandwidth(args, cdf_valid, is_dead_valid, summary_valid, src_valid=src_valid)

    print("Conditional PIT summary:", getattr(args, 'pit_cond_summary', 'rmst'))
    test_weights = _kernel_weights(summary_test, summary_valid, best_h, args.kernel)
    cdf_star = util.ecdf(t=cdf_test_all, cdf=cdf_valid, is_dead=is_dead_valid, weights=test_weights)
    end_time = time.time()

    rmst_ksp_result_test = util.metric_after_rmst_ksp(args=args, cdf=cdf_star, train_tte=tte_train, train_event=is_dead_train,
                                            tte=tte_test, is_dead=is_dead_test, order_test=order_test, src=src_test)
    print("---------------------------------------------------------------------")
    print("conditional PIT2 grid-search time:", end_time - start_time)
    print("---------------------------------------------------------------------")
    print("conditional PIT result on the test set")
    print("C-index:", rmst_ksp_result_test[0].item())
    print("S-cal(20):", rmst_ksp_result_test[1].item())
    print("D-cal(20):", rmst_ksp_result_test[2].item())
    print("KS-cal:", rmst_ksp_result_test[3].item())
    print("KM-cal:", rmst_ksp_result_test[4].item())
    print("IBS:", rmst_ksp_result_test[5].item())
    print("PSR:", rmst_ksp_result_test[6].item())
    print("Cal_ws:", rmst_ksp_result_test[7].item())

if __name__ == '__main__':
    parser = TestArgParser()
    args = parser.parse_args()

    if args.model_dist in ['cat', 'mtlr', 'psr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        # args.marginal_counts = marginal_counts

    with torch.no_grad():
        metrics = PIT_cond(args)
