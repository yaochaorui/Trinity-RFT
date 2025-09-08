# Remark: the script is for a single node training. 

cd "$(dirname "$0")/.."

config_id=${1:-1}   # default 1

# default
case $config_id in
    1)
    sync_interval=300 
    sync_offset=199
    ;;
  2)
    sync_interval=1 
    sync_offset=10
    ;;
  3)
    sync_interval=20
    sync_offset=0
    ;;  
  4)
    sync_interval=1
    sync_offset=0
    ;;  
  *)
    echo "Unknown config_id=$config_id"
    exit 1
    ;;
esac

# --- Configuration ---
project_name="sync_offset_${sync_offset}_sync_${sync_interval}"
total_steps=160
save_interval=20
eval_interval=201
random_seed=42 

mkdir -p experiments/qwen2.5-gsm8k-auto/${project_name}/
prefix=experiments/qwen2.5-gsm8k-auto/${project_name}/

zsh scripts/G1.sh $sync_interval $sync_offset $project_name $total_steps $save_interval $eval_interval $random_seed $prefix

