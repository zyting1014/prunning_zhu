MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data

MODEL=Hinge_VGG
RATIO=0.6090
TEMPLATE=CIFAR10
EPOCH=300
CHECKPOINT=${MODEL}_${TEMPLATE}_LR${EPOCH}_${RATIO}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "linear3_${TEMPLATE}_VGG" --model ${MODEL} --vgg_type 16 --batch_size 64 \
--epochs ${EPOCH} \
--teacher ${MODEL_PATH}/vgg.pt \
--pretrain ${MODEL_PATH}/vgg.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}