export MASTER_ADDR=$(hostname)  # Set master node address
export MASTER_PORT=29500         # Default PyTorch port, change if needed

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=Mamba_430M --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    pretrain.py \
    --train_data_dir /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenized_data \
    --val_data_dir /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenized_data # Change to custom train and validation data directory