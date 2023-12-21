########parameters########
lr=4e-4
lora_rank=4
lora_alpha=8
lora_trainable="query_key_value,dense"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/home2/labhosik/llama/mllama/model_output/taxi04_two_car_e32
transport_tokenizer_path=/home2/labhosik/llama/mllama/model_output/taxi04_two_car_e32
dataset_dir=/home2/labhosik/llama/mllama/04_input_data/lt_ipd
data_cache=/home2/labhosik/llama/mllama/data_chache/dc
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=32
output_dir=/home2/labhosik/llama/mllama/model_output/tllama6_lowalpha3448

deepspeed_config_file=/home2/labhosik/llama/mllama/Chinese-LLaMA-Alpaca-main/scripts/ds_zero2_no_offload.json

#######launch########
torchrun --nnodes 2 --nproc_per_node 4 /home2/labhosik/llama/mllama/Chinese-LLaMA-Alpaca-main/scripts/run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name ${transport_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --num_train_epochs 3\
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 105081 \
    --fp16 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False