# yq version 3.4.3

cd "$(dirname "$0")/../.."

# Check if correct number of arguments provided
if [ $# -ne 13 ]; then
    echo "Usage: $0 <sync_interval> <sync_offset> <project_name> <total_steps> <save_interval> <eval_interval>  <random_seed> <prefix> <config_file> <runner_num_train> <engine_num_train> <runner_num_bench> <engine_num_bench>"
    echo "Example: $0 200 100 sync_offset_100_sync_200 100 20 200 42 experiments/qwen2.5-gsm8k/sync_offset_100_sync_200/" "examples/opmd_proj/rec_gsm8k/gsm8k.yaml 64 4 128 8"
    exit 1
fi



sync_interval=$1
sync_offset=$2
project_name=$3
total_steps=$4
save_interval=$5
eval_interval=$6
random_seed=$7
prefix=$8
config_file=$9
runner_num_train=${10}
engine_num_train=${11}
runner_num_bench=${12}
engine_num_bench=${13}

prefix=$prefix$project_name/

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
    yq -i -y '.project = "'$project_name'"' "$config_file"
    yq -i -y '.synchronizer.sync_offset = '$sync_offset'' "$config_file"
    yq -i -y '.synchronizer.sync_interval = '$sync_interval'' "$config_file"
    yq -i -y '.buffer.total_steps = '$total_steps'' "$config_file"
    yq -i -y '.explorer.eval_interval = '$eval_interval'' "$config_file"
    yq -i -y '.trainer.save_interval = '$save_interval'' "$config_file"
    yq -i -y '.explorer.rollout_model.seed = '$random_seed'' "$config_file"
    
    echo "  ✓ Updated $config_file"
}

# Update config file
update_config "$config_file"


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
  runner_num=$runner_num_train
  engine_num=$engine_num_train
  echo "Mode: training and evaluation"
  # Modify the YAML file with yq
  yq -i -y \
    ".name = \"$name\" |
    .algorithm.policy_loss_fn_args.clip_mode = \"$clip_mode\" |
    .algorithm.policy_loss_fn_args.weight = \"$weight\" |
    .algorithm.advantage_fn_args.std_normalize = $std_normalize |
    .mode = \"$mode\" |
    .synchronizer.sync_method = \"$sync_method\" |
    .explorer.runner_num = $runner_num |
    .explorer.rollout_model.engine_num = $engine_num" \
    "${config_file:?config_file is not set}"

  stdbuf -oL trinity run --config "$config_file" >> "$log_file"

  mode="bench"
  sync_method="checkpoint"
  runner_num=$runner_num_bench
  engine_num=$engine_num_bench
  echo "Mode: benchmark "

    # Modify the YAML file with yq
    yq -i -y \
      ".mode = \"$mode\" |
       .synchronizer.sync_method = \"$sync_method\" |
       .explorer.runner_num = $runner_num |
       .explorer.rollout_model.engine_num = $engine_num "\
      "${config_file:?config_file is not set}"


  # # Run the experiment and log the output
  stdbuf -oL trinity run --config "$config_file" >> "$log_file"

  echo "--- Finished experiment: $name. Log saved to $log_file ---"
}

# --- Execute Experiments ---

# vanilla REINFORCE
run_experiment "REINFORCE" "none" "none" false

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

# # REC-Ring-NoIS
# run_experiment "REC-Ring-NoIS" "ring" "none" false
# rm -rf $prefix/REC-Ring-NoIS/global_step*


