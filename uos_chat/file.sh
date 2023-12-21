torchrun --nproc_per_node=4 --master_port=34322 /home2/labhosik/llama/alpaca/train.py \
    --model_name_or_path /home2/labhosik/llama/LLaMA \
    --data_path /home2/labhosik/llama/alpaca/alpaca_data.json \
    --bf16 True \
    --output_dir /home2/labhosik/llama/alpaca/alpaca_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True