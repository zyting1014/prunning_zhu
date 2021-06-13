MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data

MODEL=Prune_VGG
RATIO=0.50
TEMPLATE=CIFAR10
EPOCH=300
CHECKPOINT=${MODEL}_${TEMPLATE}_EPOCH${EPOCH}_${RATIO}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_test.py --save $CHECKPOINT --template "linear3_${TEMPLATE}_VGG" --model ${MODEL} --vgg_type 16 --batch_size 64 \
--epochs ${EPOCH} \
--teacher ${MODEL_PATH}/vgg.pt \
--pretrain ${MODEL_PATH}/vgg.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH} \
--distillation