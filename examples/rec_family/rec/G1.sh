# yq (https://github.com/mikefarah/yq/) version v4.44.2


cd "$(dirname "$0")/.."

# Check if correct number of arguments provided
if [ $# -ne 8 ]; then
    echo "Usage: $0 <sync_interval> <sync_offset> <project_name> <total_steps> <save_interval> <eval_interval>  <random_seed> <prefix>"
    echo "Example: $0 200 100 sync_offset_100_sync_200 100 20 200 42 experiments/qwen2.5-gsm8k/sync_offset_100_sync_200/"
    exit 1
fi

# List of config directories to update
CONFIG_FILE="rec_gsm8k/gsm8k.yaml"
config_dirs=(
    "rec_gsm8k"
)

sync_interval=$1
sync_offset=$2
project_name=$3
total_steps=$4
save_interval=$5
eval_interval=$6
random_seed=$7
prefix=$8

echo "  Updating all config files with:"
echo "  sync_interval: $sync_interval"
echo "  sync_offset: $sync_offset"
echo "  project_name: $project_name"
echo "  total_steps: $total_steps"
echo "  save_interval: $save_interval"
echo "  eval_interval: $eval_interval"
echo "  random_seed: $random_seed"
echo "-----------------------------------"



# Function to update a single config file
update_config() {
    local config_file=$1
    local config_name=$(basename $(dirname $config_file))
    
    echo "Updating $config_file..."
    
    # Check if file exists
    if [ ! -f "$config_file" ]; then
        echo "  Warning: $config_file not found, skipping..."
        return
    fi
    
    # Update common parameters
    yq -i '.project = "'$project_name'"' "$config_file"
    yq -i '.synchronizer.sync_offset = '$sync_offset'' "$config_file"
    yq -i '.synchronizer.sync_interval = '$sync_interval'' "$config_file"
    yq -i '.buffer.total_steps = '$total_steps'' "$config_file"
    yq -i '.explorer.eval_interval = '$eval_interval'' "$config_file"
    yq -i '.trainer.save_interval = '$save_interval'' "$config_file"
    yq -i '.explorer.rollout_model.seed = '$random_seed'' "$config_file"
    
    echo "  ✓ Updated $config_file"
}

# Update all config files
for config_dir in "${config_dirs[@]}"; do
    config_file="$config_dir/gsm8k.yaml"
    update_config "$config_file"
done



# run the experiments with the updated configs
echo "-----------------------------------"
echo "Running experiments with updated configs..."
echo "sync_interval: $sync_interval"
echo "sync_offset: $sync_offset"
echo "project_name: ${project_name}"
echo "total_steps: ${total_steps}"
echo "save_interval: ${save_interval}"
echo "eval_interval: $eval_interval"
echo "-----------------------------------"



# Create a function to run a single experiment
# Arguments: $1: Name, $2: Clip Mode, $3: Weight

# Remark: the script is for a single node training. Modify the runner_num, engine_num below and the cluster settings in the yaml file for multi-node training.

run_experiment() {
  local name="$1"
  local clip_mode="$2"
  local weight="$3"
  local std_normalize="$4"
  local log_file="${prefix}${name}/log.txt"
  mkdir -p "${prefix}${name}/"

  echo "--- Running experiment: $name ---"
  echo "--- Log will be saved to $log_file ---"
  mode="both"
  sync_method="nccl"
  runner_num=64
  engine_num=4
  echo "Mode: training and evaluation"
  # Modify the YAML file with yq
  yq -i \
    ".name = \"$name\" |
    .algorithm.policy_loss_fn_args.clip_mode = \"$clip_mode\" |
    .algorithm.policy_loss_fn_args.weight = \"$weight\" |
    .algorithm.advantage_fn_args.std_normalize = $std_normalize |
    .mode = \"$mode\" |
    .synchronizer.sync_method = \"$sync_method\" |
    .explorer.runner_num = $runner_num |
    .explorer.rollout_model.engine_num = $engine_num" \
    "${CONFIG_FILE:?CONFIG_FILE is not set}"

  stdbuf -oL trinity run --config "$CONFIG_FILE" >> "$log_file"

  mode="bench"
  sync_method="checkpoint"
  runner_num=128
  engine_num=8
  echo "Mode: benchmark "

    # Modify the YAML file with yq
    yq -i \
      ".mode = \"$mode\" |
       .synchronizer.sync_method = \"$sync_method\" |
       .explorer.runner_num = $runner_num |
       .explorer.rollout_model.engine_num = $engine_num "\
      "${CONFIG_FILE:?CONFIG_FILE is not set}"


  # # Run the experiment and log the output
  stdbuf -oL trinity run --config "$CONFIG_FILE" >> "$log_file"

  echo "--- Finished experiment: $name. Log saved to $log_file ---"
}

# --- Execute Experiments ---

yq -i \
    '.algorithm.policy_loss_fn_args.regularizer = "none" |
      .algorithm.kl_loss_fn = "k2" |
      .algorithm.kl_loss_fn_args.kl_coef = 0.0' \
    "${CONFIG_FILE:?CONFIG_FILE is not set}"

yq -i \
    '.algorithm.policy_loss_fn_args.epsilon_low = 0.2 |
    .algorithm.policy_loss_fn_args.epsilon_high = 0.2 '  \
    "${CONFIG_FILE:?CONFIG_FILE is not set}"

yq -i \
    '.algorithm.policy_loss_fn_args.epsilon_low_prime = 0.6 |
    .algorithm.policy_loss_fn_args.epsilon_high_prime = 2.0 '  \
    "${CONFIG_FILE:?CONFIG_FILE is not set}"

# vanilla REINFORCE
run_experiment "REINFORCE" "none" "none" false
rm -rf $prefix/REINFORCE/global_step*

# GRPO 
run_experiment "GRPO" "one-side" "importance_sampling" true
rm -rf $prefix/GRPO/global_step*

# REC-OneSide-IS
run_experiment "REC-OneSide-IS" "one-side" "importance_sampling" false
rm -rf $prefix/REC-OneSide-IS/global_step*

# REC-OneSide-NoIS
run_experiment "REC-OneSide-NoIS" "one-side" "none" false
rm -rf $prefix/REC-OneSide-NoIS/global_step*

# REC-TwoSide-IS
run_experiment "REC-TwoSide-IS" "two-side" "importance_sampling" false
rm -rf $prefix/REC-TwoSide-IS/global_step*

# REC-TwoSide-NoIS
run_experiment "REC-TwoSide-NoIS" "two-side" "none" false
rm -rf $prefix/REC-TwoSide-NoIS/global_step*

# REC-Ring-NoIS
run_experiment "REC-Ring-NoIS" "ring" "none" false
rm -rf $prefix/REC-Ring-NoIS/global_step*

# REC-Ring-IS
run_experiment "REC-Ring-IS" "ring" "importance_sampling" false
rm -rf $prefix/REC-Ring-IS/global_step*

echo "-----------------------------------"

yq -i \
    '.algorithm.policy_loss_fn_args.epsilon_low = 0.6 |
    .algorithm.policy_loss_fn_args.epsilon_high = 2.0 '  \
    "${CONFIG_FILE:?CONFIG_FILE is not set}}"

# REC-OneSide-IS-ring
run_experiment "REC-OneSide-IS-ring" "one-side" "importance_sampling" true
rm -rf $prefix/REC-OneSide-IS-ring/global_step*

# REC-OneSide-NoIS-ring
run_experiment "REC-OneSide-NoIS-ring" "one-side" "none" true
rm -rf $prefix/REC-OneSide-NoIS-ring/global_step*


echo "All experiments completed."