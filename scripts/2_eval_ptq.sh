#!/bin/bash

# --- Script Configuration ---
# This script runs Post-Training Quantization (PTQ) evaluation using ptq.py.
# It accepts model path and bit-widths as positional arguments.
# Optional Flags for Hadamard R4 rotation (affecting down_proj):
#  --hadamard_online   : Apply online Hadamard to ALL down_proj layers.
#  --sparse_had        : Apply online Hadamard selectively based on the JSON file
#                        specified by --layer_list_file (or defaults).
#                        Takes precedence over --hadamard_online if file exists.
#  --layer_list_file <path> : Path to the JSON file for sparse Hadamard.
#                        Defaults to "layers_to_rotate.json". Used only with --sparse_had.
#
# If no Hadamard flag is provided, no online R4 Hadamard rotation is applied.

# --- Default Values ---
HAD_ONLINE_SHELL_FLAG=0 # Track if --hadamard_online was passed
SPARSE_HAD_SHELL_FLAG=0 # Track if --sparse_had was passed
LAYER_LIST_PATH="layers_to_rotate.json" # Default path for the JSON list
R3_ONLY_SHELL_FLAG=0
OPTIMIZED_ROTATION_PATH_VALUE=""
NB_EVAL_RUNS=1

# --- Argument Parsing ---

# Capture positional arguments first
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <model_path> <w_bits> <a_bits> <kv_bits> [options...]"
  echo "Options include: --hadamard_online, --sparse_had, --layer_list_file <path>, --online_r3_only, --optimized_rotation_path <path>, --nb_eval_runs <int>"
  exit 1
fi
MODEL_PATH="$1"
W_BITS="$2"
A_BITS="$3"
KV_BITS="$4" # Assuming $4 is used for both K and V bits

# Shift away the positional arguments to parse flags
shift 4

# Parse optional flags passed TO THE SHELL SCRIPT
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --hadamard_online)
      HAD_ONLINE_SHELL_FLAG=1
      shift # consume flag
      ;;
    --sparse_had)
      SPARSE_HAD_SHELL_FLAG=1
      shift # consume flag
      ;;
    --online_r3_only) 
      R3_ONLY_SHELL_FLAG=1; 
      shift 
      ;;
    --layer_list_file)
      if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
        LAYER_LIST_PATH="$2" # Override default path
        shift 2 # consume flag and its value
      else
        echo "ERROR: --layer_list_file requires a path argument." >&2
        exit 1
      fi
      ;;
    --optimized_rotation_path) 
      if [[ -n "$2" ]] && [[ "$2" != --* ]]; then OPTIMIZED_ROTATION_PATH="$2"; shift 2;
      else echo "ERROR: --optimized_rotation_path requires path." >&2; exit 1; fi ;;
    --nb_eval_runs) 
      if [[ -n "$2" ]] && [[ "$2" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]]; then NB_EVAL_RUNS="$2"; shift 2;
      else echo "ERROR: --nb_eval_runs requires an integer value." >&2; exit 1; fi ;;
    --timing_output_path) 
      if [[ -n "$2" ]] && [[ "$2" != --* ]]; then TIMING_OUTPUT_PATH="$2"; shift 2;
      else echo "ERROR: --timing_output_path requires a path." >&2; exit 1; fi ;;
    *) # Unknown option passed to shell script
      echo "Warning: Unknown shell script option ignored: $1"
      shift
      ;;
  esac
done

# --- Build the Python Command Arguments ---
PYTHON_ARGS=""
PYTHON_ARGS+=" --input_model \"$MODEL_PATH\"" # Use captured variable, quote path
PYTHON_ARGS+=" --w_bits $W_BITS"              # Use captured variable
PYTHON_ARGS+=" --a_bits $A_BITS"              # Use captured variable
PYTHON_ARGS+=" --k_bits $KV_BITS"             # Use captured variable
PYTHON_ARGS+=" --v_bits $KV_BITS"             # Use captured variable

# Add other fixed arguments
PYTHON_ARGS+=" --do_train False --do_eval True --per_device_eval_batch_size 4"
PYTHON_ARGS+=" --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False"
PYTHON_ARGS+=" --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128"
PYTHON_ARGS+=" --rotate" # Base rotation is always needed for these modes

# --- Conditionally add optimized rotation path ---
if [[ -n "$OPTIMIZED_ROTATION_PATH" ]]; then # Only add if the variable is not empty
    PYTHON_ARGS+=" --optimized_rotation_path \"$OPTIMIZED_ROTATION_PATH\""
    echo "INFO: Using optimized rotation path: $OPTIMIZED_ROTATION_PATH"
else
    echo "INFO: No optimized rotation path provided. Using random rotation for R1/R2."
fi
# ---------------------------------------------

# Add optimized rotation path if needed
#PYTHON_ARGS+=" --optimized_rotation_path \"optimized_rotation/R_16_4_4.bin\"" # Quote path

# Add access token if needed
PYTHON_ARGS+=" --access_token \"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\"" # Quote token

PYTHON_ARGS+=" --nb_eval_runs $NB_EVAL_RUNS" # Number of times to perform evaluation (for inference time measurements)

PYTHON_ARGS+=" --timing_output_path \"$TIMING_OUTPUT_PATH\"" # Path to save timing output


# --- Logic for Hadamard Flags passed to ptq.py ---
# Priority: sparse > global > r3_only > none

HAD_ARG_FOR_PYTHON="" # Argument to potentially add for python

if [[ $SPARSE_HAD_SHELL_FLAG -eq 1 ]]; then
  echo "INFO: --sparse_had flag detected by shell script."
  # Check file using the (potentially updated) LAYER_LIST_PATH
  if [[ -f "$LAYER_LIST_PATH" ]]; then
    # Use sparse mode: pass the path to python (ensure path is quoted if it might have spaces)
    HAD_ARG_FOR_PYTHON="--selective_had_layers_path \"$LAYER_LIST_PATH\""
    echo "INFO: Layer list file found ('$LAYER_LIST_PATH'). Enabling selective Hadamard for ptq.py."
  else
    # Sparse requested but file missing: Error out
    echo "ERROR: --sparse_had flag used, but layer list file not found at '$LAYER_LIST_PATH'"
    exit 1
  fi
elif [[ $HAD_ONLINE_SHELL_FLAG -eq 1 ]]; then
  # Sparse not used, but hadamard_online was: use global hadamard
  HAD_ARG_FOR_PYTHON="--hadamard_online"
  echo "INFO: --hadamard_online flag detected (and --sparse_had not used/failed). Enabling global Hadamard for ptq.py."
elif [[ $R3_ONLY_SHELL_FLAG -eq 1 ]]; then
  HAD_ARG_FOR_PYTHON="--online_r3_only"
  echo "INFO: Shell: --online_r3_only detected. Enabling R3 ONLY for ptq.py (No R4, R3 if k<16)."
else
  # Neither flag used
  echo "INFO: No Hadamard flags detected. No online R4 Hadamard will be applied by ptq.py."
fi

# Add the determined argument (if any) to the main Python args string
if [[ -n "$HAD_ARG_FOR_PYTHON" ]]; then
    PYTHON_ARGS+=" $HAD_ARG_FOR_PYTHON"
fi
# -----------------------------------------------------------


# --- Execute the Command ---
echo "-----------------------------------------------------------------------"
echo "Executing torchrun with the following arguments for ptq.py:"
echo "$PYTHON_ARGS"
echo "-----------------------------------------------------------------------"

# Use eval to correctly handle arguments with spaces or quotes within PYTHON_ARGS
eval torchrun --nnodes=1 --nproc_per_node=1 ptq.py $PYTHON_ARGS







# ------------------------------------------------------------- #



# OUTPUT_BASE_DIR_FOR_RUN="./ptq_eval_outputs/${MODEL_BASENAME_FOR_PATH}/${RUN_CONFIG_SUFFIX}"
# SAVED_QUANTIZED_MODEL_PATH="./saved_model" # Where ptq.py will save the HF model
# LM_EVAL_RESULTS_FILE="./lm_eval_harness_results.json"




# # --- LM Evaluation Harness ---
# echo "-----------------------------------------------------------------------"
# echo "Starting LM Evaluation Harness..."
# echo "Evaluating model from: $SAVED_QUANTIZED_MODEL_PATH"
# echo "-----------------------------------------------------------------------"



# LM_EVAL_HARNESS_PATH="./lm-evaluation-harness/lm_eval/__main__.py" # Path to the main.py file of the LM Evaluation Harness

# if [ ! -f "$LM_EVAL_HARNESS_PATH" ]; then
#     echo "ERROR: LM Evaluation Harness main.py not found at $LM_EVAL_HARNESS_PATH"
#     exit 1
# fi

# PYTHON_ARGS_LMEVAL=""
# PYTHON_ARGS_LMEVAL+=" --model hf"
# PYTHON_ARGS_LMEVAL+=" --model_args pretrained=\"$SAVED_QUANTIZED_MODEL_PATH\",trust_remote_code=True,torch_dtype=\"auto\"" # auto or bfloat16
# PYTHON_ARGS_LMEVAL+=" --tasks arc_easy" #,arc_challenge,boolq,piqa,siqa,hellaswag,openbookqa,winogrande" # Your task list
# PYTHON_ARGS_LMEVAL+=" --device cuda:0"
# PYTHON_ARGS_LMEVAL+=" --batch_size auto" # Start with 1 if 'auto' has issues, e.g. --batch_size 1
# PYTHON_ARGS_LMEVAL+=" --num_fewshot 0"
# PYTHON_ARGS_LMEVAL+=" --output_path \"$LM_EVAL_RESULTS_FILE\""

# echo "Executing LM Evaluation Harness with arguments:"
# echo "$PYTHON_ARGS_LMEVAL"
# echo "-----------------------------------------------------------------------"

# python $LM_EVAL_HARNESS_PATH $PYTHON_ARGS_LMEVAL

# if [ $? -eq 0 ]; then
#   echo "LM Evaluation Harness finished successfully. Results at: $LM_EVAL_RESULTS_FILE"
# else
#   echo "ERROR: LM Evaluation Harness execution failed."
# fi
# echo "-----------------------------------------------------------------------"
# echo "Full evaluation run for configuration $RUN_CONFIG_SUFFIX complete."
# echo "Outputs stored in $OUTPUT_BASE_DIR_FOR_RUN"
# echo "-----------------------------------------------------------------------"