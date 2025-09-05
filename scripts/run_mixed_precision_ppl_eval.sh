#!/bin/bash

# CONFIG_DIR="./multi_best_configs/llama3.2-3b"

# # Check if directory exists
# [ ! -d "$CONFIG_DIR" ] && { echo "Error: Directory $CONFIG_DIR does not exist"; exit 1; }

# # Run for each JSON file
# for config_file in "$CONFIG_DIR"/*.json; do
#     [ -f "$config_file" ] || { echo "No JSON files found"; break; }
#     echo "Processing: $(basename "$config_file")"

#     torchrun --nnodes=1 --rdzv_endpoint="localhost:29501" --nproc_per_node=1 run_mixed_precision_ppl_eval.py \
#         --input_model meta-llama/Llama-3.2-3B \
#         --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
#         --nb_eval_runs 1 \
#         --do_train False --do_eval True \
#         --per_device_eval_batch_size 4 --model_max_length 2048 \
#         --fp16 False --bf16 True --save_safetensors False \
#         --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate --eval_dataset wikitext2 --high_precision_bits 16 \
#         --mixed_precision_config "$config_file" \
#         --load_qmodel_path "quantized_weights_offline_rotated_llama-3.2-3b.pt" \
#         --block_size 32
# done

torchrun --nnodes=1 --rdzv_endpoint="localhost:29504" --nproc_per_node=1 run_mixed_precision_ppl_eval.py \
        --input_model meta-llama/Llama-3.2-3B \
        --w_bits 4 --a_bits 16 --k_bits 16 --v_bits 16 \
        --nb_eval_runs 1 \
        --do_train False --do_eval True \
        --per_device_eval_batch_size 4 --model_max_length 2048 \
        --fp16 False --bf16 True --save_safetensors False \
        --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate --eval_dataset wikitext2 \
        --load_qmodel_path "quantized_weights_offline_rotated_llama-3.2-3b.pt" \