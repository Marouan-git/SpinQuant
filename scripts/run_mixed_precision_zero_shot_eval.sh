#!/bin/bash

CONFIG_DIR="./multi_best_configs/llama2-7b/topk/k5/module_wise"

# Check if directory exists
[ ! -d "$CONFIG_DIR" ] && { echo "Error: Directory $CONFIG_DIR does not exist"; exit 1; }

# Run for each JSON file
for config_file in "$CONFIG_DIR"/*.json; do
    [ -f "$config_file" ] || { echo "No JSON files found"; break; }
    
    echo "Processing: $(basename "$config_file")"
    
    torchrun --nnodes=1 --rdzv_endpoint="localhost:29502" --nproc_per_node=1 run_mixed_precision_zero_shot_eval.py \
        --input_model meta-llama/Llama-2-7b-hf \
        --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
        --nb_eval_runs 1 \
        --optimized_rotation_path optimized_rotation/R_16_4_4.bin \
        --do_train False --do_eval True \
        --per_device_eval_batch_size 4 --model_max_length 2048 \
        --fp16 False --bf16 True --save_safetensors False \
        --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate --eval_dataset wikitext2 --high_precision_bits 16 \
        --mixed_precision_config "$config_file" \
        --access_token "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" \
        --load_qmodel_path "quantized_weights_offline_rotated.pt" \
        --block_size 64
done

# torchrun --nnodes=1 --rdzv_endpoint="localhost:29501" --nproc_per_node=1 run_mixed_precision_zero_shot_eval.py \
#         --input_model meta-llama/Llama-2-7b-hf \
#         --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
#         --nb_eval_runs 1 \
#         --optimized_rotation_path optimized_rotation/R_16_4_4.bin \
#         --do_train False --do_eval True \
#         --per_device_eval_batch_size 4 --model_max_length 2048 \
#         --fp16 False --bf16 True --save_safetensors False \
#         --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate --eval_dataset c4 --high_precision_bits 16 \
#         --access_token "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"