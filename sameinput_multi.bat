@REM nohup python main_50.py --model VAE-RNN --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VAE-RNN-B.txt   
@REM nohup python main_50.py --model VAE-RNN --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VAE-RNN-None.txt  &
@REM nohup python main_50.py   --model VRNN-Gauss-I --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-I-B.txt &
@REM nohup python main_50.py   --model VRNN-Gauss-I --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-I-None.txt 
@REM nohup python main_50.py --model VRNN-Gauss --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 2  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-B.txt  
@REM nohup python main_50.py --model VRNN-Gauss --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 2  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-None.txt  
@REM nohup python main_50.py   --model VRNN-GMM-I --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 70 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-I-B.txt 
@REM nohup python main_50.py   --model VRNN-GMM-I --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 70 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-I-None.txt 
@REM nohup python main_50.py --model VRNN-GMM --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-B.txt  
@REM nohup python main_50.py --model VRNN-GMM --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-None.txt  
@REM nohup python main_50.py  --model STORN --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 60 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/STORN-B.txt   
@REM nohup python main_50.py  --model STORN --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 60 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/STORN-None.txt   

@REM nohup python main_50.py --model VRNN-GMM --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-None.txt  &
@REM nohup python main_50.py  --model STORN --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 60 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/STORN-None.txt   &

nohup python main_single.py --h_dim 2 &
nohup python main_single.py --h_dim 5 &
nohup python main_single.py --h_dim 10 &
nohup python main_single.py --h_dim 20 &
nohup python main_single.py --h_dim 30 &
nohup python main_single.py --h_dim 40 &
nohup python main_single.py --h_dim 50 &
nohup python main_single.py --h_dim 60 &
nohup python main_single.py --h_dim 80 &

@REM python main_single.py --model "VRNN-Gauss" --known_parameter "None" --train_rounds 2 --logdir "multi_50"
@REM python main_single.py --model "VRNN-Gauss" --known_parameter "B"
@REM python main_single.py --model "VRNN-Gauss-I" --known_parameter "None"
@REM python main_single.py --model "VRNN-Gauss-I" --known_parameter "B"
@REM python main_single.py --model "VRNN-GMM" --known_parameter "None"
@REM python main_single.py --model "VRNN-GMM" --known_parameter "B"
@REM python main_single.py --model "VRNN-GMM-I" --known_parameter "None"
@REM python main_single.py --model "VRNN-GMM-I" --known_parameter "B"
@REM python main_single.py --model "STORN" --known_parameter "None"
@REM python main_single.py --model "STORN" --known_parameter "B"

@REM nohup python main_50.py --model VAE-RNN --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VAE-RNN-B.txt   2>&1
@REM nohup python main_50.py --model VAE-RNN --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VAE-RNN-None.txt  2>&1
@REM nohup python main_50.py   --model VRNN-Gauss-I --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-I-B.txt 2>&1
@REM nohup python main_50.py   --model VRNN-Gauss-I --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-I-None.txt 2>&1
@REM nohup python main_50.py --model VRNN-Gauss --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 2  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-B.txt  2>&1
@REM nohup python main_50.py --model VRNN-Gauss --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 80 --z_dim 2  > ./log/multi_correct_para/toy_lgssm/VRNN-Gauss-None.txt  2>&1
@REM nohup python main_50.py   --model VRNN-GMM-I --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 70 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-I-B.txt 2>&1
@REM nohup python main_50.py   --model VRNN-GMM-I --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 70 --z_dim 10  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-I-None.txt 2>&1
@REM nohup python main_50.py --model VRNN-GMM --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-B.txt  2>&1
@REM nohup python main_50.py --model VRNN-GMM --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 50 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/VRNN-GMM-None.txt  2>&1
@REM nohup python main_50.py  --model STORN --known_parameter B --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 60 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/STORN-B.txt   2>&1
@REM nohup python main_50.py  --model STORN --known_parameter None --train_rounds 50 --logdir "multi_correct_para"   --do_test "True" --do_train "True" --h_dim 60 --z_dim 5  > ./log/multi_correct_para/toy_lgssm/STORN-None.txt   2>&1
