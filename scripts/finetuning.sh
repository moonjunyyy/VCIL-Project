#!/bin/bash

#SBATCH --job-name Finetuning_iblurry_cifar100_N50_M10
#SBATCH -p batch
#SBATCH -w vll4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=4-0
#SBATCH -o %x_%j.log


date
# seeds=(1 21 42 3473 10741 32450 93462 85015 64648 71950 87557 99668 55552 4811 10741)
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /data/keonhee/init.sh
conda activate torch38gpu

conda --version
python --version
echo "Batch size 32 onlin iter 3"
# CIL CONFIG
MODE="Finetuning"
NOTE="Finetuning_iblurry_cifar100_N50_M10_gpu2" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3"
VIT="True"
OPT="adam"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="vit" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME=$OPT SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=32; LR=3e-4 OPT_NAME=$OPT SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=100
    BATCHSIZE=32; LR= OPT_NAME=$OPT SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=256; LR= OPT_NAME=$OPT SCHED_NAME="multistep" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

if [ "$VIT" == "True" ]; then
    echo "Vit is used"
    MODEL_NAME="vit"
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --n_worker 4 \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /local_datasets/ \
    --note $NOTE --eval_period $EVAL_PERIOD --memory_epoch $MEMORY_EPOCH --n_worker 4
done