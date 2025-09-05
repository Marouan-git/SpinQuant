export MASTER_PORT=29504

torchrun --nnodes=1 --rdzv_endpoint="localhost:29503" --nproc_per_node=1 run_ppl_module_analysis.py \
    --input_model meta-llama/Llama-3.2-3B \
    --w_bits 4 \
    --a_bits 8 \
    --k_bits 8 \
    --v_bits 8 \
    --nb_eval_runs 1 \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate \
    --load_qmodel_path "quantized_weights_offline_rotated_llama-3.2-3b.pt" \
    #--save_qmodel_path "quantized_weights_offline_rotated_llama-3-8b.pt" \