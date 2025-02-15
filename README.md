```bash
python scripts/data_preparation.py --raw_data_dir data/raw --processed_data_dir data/processed
```
CUDA_VISIBLE_DEVICES=2 python scripts/compute_activations.py \
  --data_dir ./data/processed \
  --model_root_path /mnt/data102_d2/huggingface/models \
  --model_name Llama-2-7b-hf \
  --activations_root_path ./activations \
  --sample_size 200 \
  --min_shot 0 \
  --max_shot 1 \
  --shot_seed 44 \
  --seed 42

CUDA_VISIBLE_DEVICES=4 python scripts/compute_pruning_indices.py \
  --activations_root_path ./activations \
  --model_name Llama-2-7b-hf \
  --task_types mmlu hellaswag piqa gsm8k ai2_arc winogrande \
  --global_sparsity 0.1 \
  --x0 0.5 \
  --k_base 10 \
  --low_val_others 0.7 \
  --high_val_others 1.0 \
  --x1_others 0.7 \
  --k1_others 5 \
  --m 3 \
  --output_dir ./pruned_indices \
  --seed 42

CUDA_VISIBLE_DEVICES=6,7 python evaluate_multiple_tasks.py \
    --model_root_path /mnt/data102_d2/huggingface/models \
    --model_name Llama-2-7b-hf \
    --pruning_indices_root_dir ./pruned_indices \
    --global_sparsity 0.2 \
    --temp_dir ./tmp \
    --output_base_dir ./eval_out \
    --use_cache_base_dir ./eval_cache \
    --keep_temp

CUDA_VISIBLE_DEVICES=6,7 python scripts/prune_model.py \
  --model_root_path /mnt/data102_d2/huggingface/models \
  --model_name Llama-2-7b-hf \
  --pruning_indices_root_dir "./pruned_indices" \
  --task_types mmlu piqa \
  --global_sparsity 0.2 \
  --output_dir ./pruned_models