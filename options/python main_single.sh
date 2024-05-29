# baseline model
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=80 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=80 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=20 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=20 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=40 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=40 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=60 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=60 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=100 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=100 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    # have not been trained
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=120 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=120 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=10 --z_dim=10  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=10 --z_dim=5  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

# hard constraints: 


    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_hc_relu'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &


# phy+aug constraints: 

    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='VAE-RNN-PHYNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &

python main_single.py --n_epoch=5 --model='VAE-RNN-PHYNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_phy_aug'  --mpnt_wt=1000  --do_test --do_train --train_rounds=2

#  not variance inference:
  nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout'  --mpnt_wt=0  --do_test --do_train --train_rounds=5 &

# hard constraint
  nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_hc'  --mpnt_wt=1000  --do_test --do_train --train_rounds=5 &


# sft constraint
  nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=120 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=100 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=80 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=60 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=40 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=20 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
    nohup python main_single.py --n_epoch=750 --model='AE-RNN' --h_dim=10 --z_dim=2  --dataset='toy_lgssm_5_pre' --logdir='baseline_test_rnn_nodropout_sc'  --mpnt_wt=10  --do_test --do_train --train_rounds=5 &
