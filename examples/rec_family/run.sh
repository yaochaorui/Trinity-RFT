#!/bin/bash
# filepath: ~/Trinity-RFT-dev/examples/rec_family/run.sh

cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

### config the parameters below ###

project_name='sync_offset_'$sync_offset'_sync_'$sync_interval
total_steps=2000
save_interval=100
eval_interval=2001
random_seed=42

config_id=${1:-1}   # default 1

# default
case $config_id in
  1)
    sync_interval=1 
    sync_offset=0
    ;;
  2)
    sync_interval=250 
    sync_offset=0
    ;;
  3)
    sync_interval=2000
    sync_offset=1999
    ;;
  *)
    echo "Unknown config_id=$config_id"
    exit 1
    ;;
esac


# config this according to your hardware. Default is for 8 L20 GPUs.
runner_num_train=64
engine_num_train=4
runner_num_bench=128
engine_num_bench=8


# qwen+gsm8k
# CONFIG_FILE="examples/rec_gsm8k/gsm8k.yaml"
# model_path='You model path here'


# llama+math


CONFIG_FILE="$(pwd)/examples/rec_math/math.yaml"
model_path='your model path here'

# modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir $DATASET_PATH/{dataset_name}
train_data_path='$DATASET_PATH/{dataset_name}/data'
# here we only evaluate one dataset, you can add more eval datasets in the yaml file if needed.
eval_data_path='$DATASET_PATH/{dataset_name}/data'

exp_name='llama-math-G1' 

##################################


mkdir -p $(pwd)/experiments/${exp_name}/${project_name}/
prefix=$(pwd)/experiments/${exp_name}/
checkpoint_root_dir=$prefix

yq -i -y '.model.model_path = "'$model_path'"' $CONFIG_FILE
yq -i -y '.checkpoint_root_dir = "'$checkpoint_root_dir'"' $CONFIG_FILE
yq -i -y '.buffer.explorer_input.taskset.path = "'$train_data_path'"' $CONFIG_FILE
yq -i -y '.buffer.explorer_input.eval_tasksets[0].path = "'$eval_data_path'"' $CONFIG_FILE
zsh examples/rec_family/G1.sh $sync_interval $sync_offset $project_name $total_steps $save_interval $eval_interval $mode $random_seed $prefix $CONFIG_FILE $runner_num_train $engine_num_train $runner_num_bench $engine_num_bench

