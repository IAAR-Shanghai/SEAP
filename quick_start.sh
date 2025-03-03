#!/bin/bash

# A simple script to run the entire pipeline

echo "Starting data preparation..."

# Step 1: Data preparation
python scripts/data_preparation.py \
  --raw_data_dir data/raw \
  --processed_data_dir data/processed

if [ $? -ne 0 ]; then
  echo "Error in data preparation. Exiting..."
  exit 1
fi

echo "Data preparation completed."

# Step 2: Compute activations
echo "Computing activations..."

python scripts/compute_activations.py \
  --data_dir ./data/processed \
  --model_root_path ../models \
  --model_name Llama-2-7b-hf \
  --activations_root_path ./activations

if [ $? -ne 0 ]; then
  echo "Error in computing activations. Exiting..."
  exit 1
fi

echo "Activations computation completed."

# Step 3: Compute masks
echo "Computing pruning masks..."

python scripts/compute_masks.py \
  --model_root_path ../models \
  --model_name Llama-2-7b-hf \
  --activations_root_path ./activations \
  --output_dir ./pruning_masks \
  --method WIFV \
  --structure UL-LD \
  --pruning_ratio 0.2 \
  --use_generic_mask

if [ $? -ne 0 ]; then
  echo "Error in computing pruning masks. Exiting..."
  exit 1
fi

echo "Pruning masks computation completed."

# Step 4: Apply pruning
echo "Applying pruning..."

python scripts/apply_pruning.py \
  --model_root_path ../models \
  --model_name Llama-2-7b-hf \
  --masks_root_dir ./pruning_masks \
  --task_types gsm8k \
  --output_dir ./pruned_models \
  --softmask \
  --pruning_ratio 0.2 \
  --activations_root_path ./activations

if [ $? -ne 0 ]; then
  echo "Error in applying pruning. Exiting..."
  exit 1
fi

echo "Pruning application completed."

# Step 5: Evaluate the pruned model
echo "Evaluating pruned model..."

python evaluate_multiple_tasks.py \
  --model_root_path ../models \
  --model_name Llama-2-7b-hf \
  --pruning_indices_root_dir ./pruning_masks \
  --pruning_ratio 0.2 \
  --temp_dir ./pruned_models \
  --output_base_dir ./eval_out \
  --softmask \
  --use_wiki_mask \
  --use_generic_mask

if [ $? -ne 0 ]; then
  echo "Error in evaluating pruned model. Exiting..."
  exit 1
fi

echo "Evaluation completed successfully!"
