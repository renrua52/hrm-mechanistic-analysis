#!/bin/bash

# Example script to run parallel batch inference
# This script demonstrates how to use parallel_batch_inference.py with various options

# Define checkpoints (comma-separated list)
CHECKPOINTS="checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint"

# Define number of permutations
PERMUTES=5

# Define maximum number of worker threads (adjust based on your system capabilities)
MAX_WORKERS=8

# Define output file for results
OUTPUT_FILE="inference_results_$(date +%Y%m%d_%H%M%S).json"

# Run the parallel inference script
echo "Starting parallel inference with $PERMUTES permutations and $MAX_WORKERS workers..."
python parallel_batch_inference.py \
  --checkpoints "$CHECKPOINTS" \
  --permutes $PERMUTES \
  --max_workers $MAX_WORKERS \
  --output "$OUTPUT_FILE" \
  --num_batch 182 \
  --batch_size 2323

echo "Inference completed. Results saved to $OUTPUT_FILE"
