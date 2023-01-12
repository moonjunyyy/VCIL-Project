
conda --version
python --version

source /data/keonhee/init.sh
conda activate torch38gpu

# CIL CONFIG
MODE="Finetuning"
NOTE="Finetuning_iblurry_cifar100_N50_M10_test" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3"
# VIT="True"
OPT="adam"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=128; LR=0.03 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=100
    BATCHSIZE=16; LR=0.001 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=100
    BATCHSIZE=32; LR=0.05 OPT_NAME=$OPT SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=0.05 OPT_NAME=$OPT SCHED_NAME="multistep" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

# if [ "$VIT" == "True" ]; then
#     echo "Vit is used"
#     MODEL_NAME="vit"
# fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /local_datasets/ \
    --note $NOTE --eval_period $EVAL_PERIOD --memory_epoch $MEMORY_EPOCH $USE_AMP --debug
done