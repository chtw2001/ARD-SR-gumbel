# sh for runpod
timestamp=$(date +'%y_%m_%d_%H_%M_%S')
logfile=./logs/MHCN_$timestamp.log

sweep_output=$(wandb sweep sweep.yaml 2>&1)
sweep_path=$(echo "$sweep_output" | grep "wandb agent" | awk '{print $NF}')
echo "Sweep path: $sweep_path"

PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
nohup wandb agent $sweep_path > $logfile 2>&1 &