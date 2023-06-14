#!/bin/bash

learning_rate=$1
latent_dim=$2
batch_size=$3

PARENT_DIR="$(dirname $PWD)"
EXEC_DIR=$PWD
log_dir="logs/lr=${learning_rate}_latentdim=${latent_dim}_batchsize=${batch_size}"
mkdir -p $log_dir
# export run_exec=$PARENT_DIR/train.py
# # export run_exec=/afs/crc.nd.edu/user/k/kkosaraj/kristools/microgrid_dcbf.py
# export run_flags="--learning-rate=${learning_rate} --latent-dim=${latent_dim} --batch-size=${batch_size} --log_dir='${log_dir}' "  
echo "Current working directory is: $(pwd)"
python train.py --image-dir='../train_data' --learning-rate=${learning_rate} --latent-dim=${latent_dim} --batch-size=${batch_size} --logs-dir=${log_dir}