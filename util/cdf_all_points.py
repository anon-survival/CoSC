import torch
import util
from saver import ModelSaver
from lifelines.utils import concordance_index
from util.postprocessing import safe_logit, normalized_rmst
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cdf_all_points(args, phase='test'):
    if args.model_dist in ['cat', 'mtlr', 'psr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        # args.marginal_counts = marginal_counts

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    if phase == 'valid':
        test_loader, _ = eval_loaders
    elif phase == 'test':
        _, test_loader = eval_loaders
    else:
        assert False

    # args.loss_fn = 'mle'

    for src_test, tgt_test in test_loader:
        src_test = src_test.to(DEVICE)
        tgt_test = tgt_test.to(DEVICE)
        tte_test = tgt_test[:, 0]
        is_dead_test = tgt_test[:, 1]
        order_test = torch.argsort(tte_test)

    pred_params_test = model.forward(src_test)
    if args.model_dist in ['cat', 'mtlr', 'psr']:
        tgt_test = util.cat_bin_target(args, tgt_test, args.bin_boundaries)
        
    cdf_test = util.get_cdf_matrix(pred_params_test, tgt_test, args)
    if args.model_dist != 'cox':
        cdf_test = cdf_test[:, order_test][order_test, :]

    else:
        cdf_test = cdf_test

    tte_test = tte_test[order_test]
    is_dead_test = is_dead_test[order_test]
    src_test = src_test[order_test]

    return cdf_test, tte_test, is_dead_test, src_test, order_test

def metric_after_ksp(args, cdf, train_tte, train_event, tte, is_dead, order_test, src, a, b, alpha):
    with torch.no_grad():
        cdf = torch.sigmoid(a * safe_logit(cdf.float()) + b) ** alpha

    cdf_diag = torch.diag(cdf)

    # calculate the mean
    surv = 1 - cdf
    surv = torch.concat([torch.ones((surv.shape[0], 1)).to(DEVICE), surv], dim=1)

    tte2 = tte.unsqueeze(0).repeat(tte.shape[0], 1)
    tte2 = torch.concat([torch.zeros(tte2.shape[0], 1).to(DEVICE), tte2], dim=1)

    last_element = torch.where(surv[:, -1] != 0, tte2[:, -1]/(1-surv[:, -1]), tte2[:, -1]).unsqueeze(1)

    tte2 = torch.cat([tte2, last_element], dim=1)

    surv = torch.concat([surv, torch.zeros((surv.shape[0], 1)).to(DEVICE)], dim=1)
    batch_size = 5000
    integral = torch.zeros(tte.shape).to(DEVICE)
    for i in range(0, surv.shape[0], batch_size):
        integral[i:i+batch_size] = torch.trapezoid(surv[i:i+batch_size, :], tte2[i:i+batch_size, :])

    # if args.dataset in ['liver', 'stomach', 'bladder']:
    #     labels_test = torch.load(f"./data/seer/{args.dataset}/{args.k}/{args.dataset}_test_labels.pt").to(DEVICE)
    #     labels_test = labels_test[order_test]
    # else:
    #     labels_test = torch.load(f"./data/{args.dataset}/{args.k}/{args.dataset}_test_labels.pt").to(DEVICE)
    #     labels_test = labels_test[order_test]

    # KS_SUM, KS_VAR = groupwise_ks_metric(cdf_diag, is_dead, labels_test)

    C_index = concordance_index(tte.cpu(), integral.cpu(), is_dead.cpu())
    SCAL = util.s_calibration(points=cdf_diag, is_dead=is_dead, phase='test', args=args, device=DEVICE)
    DCAL = util.d_calibration(points=cdf_diag, is_dead=is_dead, args=args, phase='test', device=DEVICE)
    _, KS = util.get_p_value(args=args, cdf=cdf_diag, is_dead=is_dead, device=DEVICE)
    KM_CAL = util.km_calibration(cdf=cdf, tte=tte, is_dead=is_dead, device=DEVICE)
    IBS = util.integrated_brier_score(train_tte=train_tte, train_event=train_event,
                                      test_tte=tte, test_event=is_dead, cdf_test=cdf, time=tte)
    PSR = util.cen_log_simple(tte=tte, is_dead=is_dead, cdf_matrix=cdf)
    cal_ws = util.wsc_xcal(X=src, event_indicators=is_dead, predict_probs=1-cdf_diag)

    return C_index, SCAL, DCAL, KS, KM_CAL, IBS, PSR, cal_ws

def metric_after_rmst_ksp(args, cdf, train_tte, train_event, tte, is_dead, order_test, src):
    cdf_diag = torch.diag(cdf)

    # calculate the mean
    surv = 1 - cdf
    surv = torch.concat([torch.ones((surv.shape[0], 1)).to(DEVICE), surv], dim=1)

    tte2 = tte.unsqueeze(0).repeat(tte.shape[0], 1)
    tte2 = torch.concat([torch.zeros(tte2.shape[0], 1).to(DEVICE), tte2], dim=1)

    last_element = torch.where(surv[:, -1] != 0, tte2[:, -1]/(1-surv[:, -1]), tte2[:, -1]).unsqueeze(1)

    tte2 = torch.cat([tte2, last_element], dim=1)

    surv = torch.concat([surv, torch.zeros((surv.shape[0], 1)).to(DEVICE)], dim=1)
    batch_size = 5000
    integral = torch.zeros(tte.shape).to(DEVICE)
    for i in range(0, surv.shape[0], batch_size):
        integral[i:i+batch_size] = torch.trapezoid(surv[i:i+batch_size, :], tte2[i:i+batch_size, :])

    # if args.dataset in ['liver', 'stomach', 'bladder']:
    #     labels_test = torch.load(f"./data/seer/{args.dataset}/{args.k}/{args.dataset}_test_labels.pt").to(DEVICE)
    #     labels_test = labels_test[order_test]
    # else:
    #     labels_test = torch.load(f"./data/{args.dataset}/{args.k}/{args.dataset}_test_labels.pt").to(DEVICE)
    #     labels_test = labels_test[order_test]
        
    # KS_SUM, KS_VAR = groupwise_ks_metric(cdf_diag, is_dead, labels_test)

    C_index = concordance_index(tte.cpu(), integral.cpu(), is_dead.cpu())
    SCAL = util.s_calibration(points=cdf_diag, is_dead=is_dead, phase='test', args=args, device=DEVICE)
    DCAL = util.d_calibration(points=cdf_diag, is_dead=is_dead, args=args, phase='test', device=DEVICE)
    _, KS = util.get_p_value(args=args, cdf=cdf_diag, is_dead=is_dead, device=DEVICE)
    KM_CAL = util.km_calibration(cdf=cdf, tte=tte, is_dead=is_dead, device=DEVICE)
    IBS = util.integrated_brier_score(train_tte=train_tte, train_event=train_event,
                                      test_tte=tte, test_event=is_dead, cdf_test=cdf, time=tte)
    PSR = util.cen_log_simple(tte=tte, is_dead=is_dead, cdf_matrix=cdf)
    cal_ws = util.wsc_xcal(X=src, event_indicators=is_dead, predict_probs=1-cdf_diag)

    return C_index, SCAL, DCAL, KS, KM_CAL, IBS, PSR, cal_ws

def groupwise_ks_metric(cdf, is_dead, labels_test):
    EPS = 1e-8

    cdf_order = torch.argsort(cdf)
    F_sorted = cdf[cdf_order].unsqueeze(1)
    is_dead = is_dead[cdf_order].unsqueeze(1)
    labels_test = labels_test[cdf_order].unsqueeze(1)

    K = len(torch.unique(labels_test))
    N = cdf.shape[0]
    
    KS = torch.zeros(K, device=DEVICE)
    for k in range(K):
        Nj = (labels_test == k).sum()

        is_alive_k = (1 - is_dead).float()

        denom_k = 1 - F_sorted + EPS
        weight_k = is_alive_k / denom_k
        F_weight_k = F_sorted * weight_k

        cum_weight_k = torch.cumsum(weight_k * (labels_test == k), dim=0)
        cum_F_weight_k = torch.cumsum(F_weight_k * (labels_test == k), dim=0)

        cum_weight_shifted_k = F.pad(cum_weight_k[:-1], (0, 0, 1, 0), value=0.0)
        cum_F_weight_shifted_k = F.pad(cum_F_weight_k[:-1], (0, 0, 1, 0), value=0.0)

        ecdf_cens_k = F_sorted * cum_weight_shifted_k - cum_F_weight_shifted_k
        ecdf_cens_k = torch.clamp(ecdf_cens_k, 0, Nj)

        ecdf_dead_k = torch.cumsum(is_dead * (labels_test == k), dim=0)
        ecdf_upper_k = (ecdf_dead_k + ecdf_cens_k) / N
        ecdf_upper_k = torch.clamp(ecdf_upper_k, 0, Nj/N)
        ecdf_lower_k = ecdf_upper_k - is_dead * (labels_test == k) / N

        KS_upper_k = torch.abs(ecdf_upper_k - F_sorted*(Nj/N))
        KS_lower_k = torch.abs(ecdf_lower_k - F_sorted*(Nj/N))
        KS_error_k = torch.maximum(KS_upper_k, KS_lower_k)

        KS[k] = torch.max(KS_error_k)
    
    # KS_max = KS.max()
    KS_sum = KS.sum()
    KS_var = KS.var()

    return KS_sum, KS_var
