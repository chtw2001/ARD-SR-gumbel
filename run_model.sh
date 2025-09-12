timestamp=$(date +'%y_%m_%d_%H_%M_%S')
logfile=./logs/ARD-SR_gumbel_$timestamp.log

sweep_output=$(WANDB_API_KEY=$WANDB_API_KEY /home/taekwon/.conda/envs/ARD-SR-debug/bin/wandb sweep sweep.yaml 2>&1)
sweep_path=$(echo "$sweep_output" | grep "wandb agent" | awk '{print $NF}')
echo "Sweep path: $sweep_path"

WANDB_API_KEY=$WANDB_API_KEY \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
nohup /home/taekwon/.conda/envs/ARD-SR-debug/bin/wandb agent $sweep_path > $logfile 2>&1 &