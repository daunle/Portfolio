pip install openai
pip install accelerate -U
huggingface-cli login --token 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
wandb login 0e4dc0c692d80321fae8974d20c7c5018b1bd413
torchrun --nproc_per_node=4 --master_port=36725 /home2/labhosik/llama/alpaca/train.py\
    --model_name_or_path meta-llama/Llama-2-7b \
    --data_path /home2/labhosik/llama/experiment/tips/general_dataset/general_dataset.json \
    --bf16 True \
    --output_dir /home2/labhosik/llama/experiment/tips/output_model/test_model_general_dataset   \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "/home2/labhosik/llama/alpaca//configs/default_offload_opt_param.json" \
    --tf32 True