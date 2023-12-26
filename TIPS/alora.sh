wandb login 0e4dc0c692d80321fae8974d20c7c5018b1bd413
python3 /home2/labhosik/llama/alpaca-lora-main/finetune_pg.py \
    --base_model '/home2/labhosik/llama/mllama/model_output/fin_model' \
    --data_path '/home2/labhosik/llama/data.json' \
    --output_dir '/home2/labhosik/llama/mllama/model_output/lora-alpaca/munza_2' \
    --batch_size 2048 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ['query_key_value'] \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint /home2/labhosik/llama/mllama/model_output/lora-alpaca/munza_2/checkpoint-400