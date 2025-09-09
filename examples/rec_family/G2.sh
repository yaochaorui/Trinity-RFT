# yq version 3.4.3

cd "$(dirname "$0")/.."

# Check if correct number of arguments provided
if [ $# -ne 9 ]; then
    echo "Usage: $0 <sync_interval> <sync_offset> <project_name> <total_steps> <save_interval> <eval_interval>  <random_seed> <prefix> <config_file>"
    echo "Example: $0 200 100 sync_offset_100_sync_200 100 20 200 42 experiments/qwen2.5-gsm8k/sync_offset_100_sync_200/" "examples/opmd_proj/rec_gsm8k/gsm8k.yaml"
    exit 1
fi

# List of config directories to update
config_file=$9

sync_interval=$1
sync_offset=$2
project_name=$3
total_steps=$4
save_interval=$5
eval_interval=$6
random_seed=$7
prefix=$8

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

# Update the config file
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

run_experiment() {
  local name="$1"
  local clip_mode="$2"
  local weight="$3"
  local std_normalize="$4"
  local regularizer="$5"
  local regularizer_coef="$6"
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
  yq -i -y \
    ".name = \"$name\" |
    .algorithm.policy_loss_fn_args.clip_mode = \"$clip_mode\" |
    .algorithm.policy_loss_fn_args.weight = \"$weight\" |
    .algorithm.advantage_fn_args.std_normalize = $std_normalize |
    .algorithm.policy_loss_fn_args.regularizer = \"$regularizer\" |
    .algorithm.policy_loss_fn_args.regularizer_coef = $regularizer_coef |
    .mode = \"$mode\" |
    .synchronizer.sync_method = \"$sync_method\" |
    .explorer.runner_num = $runner_num |
    .explorer.rollout_model.engine_num = $engine_num" \
    "${config_file:?config_file is not set}"

  stdbuf -oL trinity run --config "$config_file" >> "$log_file"

  mode="bench"
  sync_method="checkpoint"
  runner_num=128
  engine_num=8
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

yq -i -y \
    '.algorithm.policy_loss_fn_args.regularizer = "none" |
      .algorithm.kl_loss_fn = "k2" |
      .algorithm.kl_loss_fn_args.kl_coef = 0.0' \
    "${config_file:?config_file is not set}"
    
# AsymRE
run_experiment "AsymRE" "none" "none" false "forward-kl" 0.2
rm -rf $prefix/AsymRE/global_step*

# # OPMD
# run_experiment "OPMD" "none" "none" false "k2" 0.1
# rm -rf $prefix/OPMD/global_step*
