#!/bin/bash


#SBATCH -J RM_test_new_Adam_iblurry_cifar100
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -w vll1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=4-0
#SBATCH -o %x_%j.out


source /data/keonhee/init.sh
conda activate torch38gpu

# CIL CONFIG
NOTE="[rm-test]_iblurry_cifar100_3epoch" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="rm"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=10
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
# USE_AMP="--use_amp"
SEEDS="1 2 3"
OPT="adam"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=0.05 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=16; LR=0.05 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=32; LR=0.05 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=0.05 OPT_NAME=$OPT SCHED_NAME="multistep" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /local_datasets/VCIL/$DATASET \
    --note $NOTE --eval_period $EVAL_PERIOD --memory_epoch $MEMORY_EPOCH $USE_AMP
done
