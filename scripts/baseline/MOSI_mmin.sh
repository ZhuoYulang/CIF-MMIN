set -e
run_idx=$1
gpu=$2

for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=mmin 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=$gpu
--A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=visual --input_dim_v=47 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=768 --embd_size_l=128
--AE_layers=256,128,64 --n_blocks=5 --share_weight
--pretrained_path=checkpoints/MOSI_utt_fusion_AVL_run1
--num_thread=8
--ce_weight=1.0 --mse_weight=2.0 --cycle_weight=0.5
--output_dim=1 --cls_layers=128,64 --dropout_rate=0.5
--niter=30 --niter_decay=10 --verbose --weight_decay=1e-5
--batch_size=64 --lr=2e-4 --run_idx=$run_idx --corpus_name=MOSI
--name=mmin_MOSI_f1 --suffix=block_{n_blocks}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done