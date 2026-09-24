First, run train.py to train the model.

# DeepSurv

python train.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 50 --model SyntheticNN --model_dist cox --dropout_rate 0.1 --num_epochs 1000

And then, run the code for each methods.

# Baseline

python test.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 982 --phase test --model SyntheticNN --model_dist cox --dropout_rate 0.0 --ckpt_path ckpts/whas/DeepSurv/(your checkpoint name)/

# KSP

python KSP.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 982 --phase test --model SyntheticNN --model_dist cox --dropout_rate 0.0 --ckpt_path ckpts/whas/DeepSurv/(your checkpoint name)/

# CSD 
python CSD.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 982 --phase test --model SyntheticNN --model_dist cox --dropout_rate 0.0 --ckpt_path ckpts/whas/DeepSurv/(your checkpoint name)/

# CSD-iPOT

python CSD-iPOT.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 982 --phase test --model SyntheticNN --model_dist cox --dropout_rate 0.0 --ckpt_path ckpts/whas/DeepSurv/(your checkpoint name)/

# CoSC

python CoSC.py --name ks --dataset whas --data_dir data/whas/1/ --batch_size 982 --phase test --model SyntheticNN --model_dist cox --dropout_rate 0.0 --ckpt_path ckpts/whas/DeepSurv/(your checkpoint name)/

