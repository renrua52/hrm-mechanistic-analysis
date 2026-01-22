#!/bin/bash

# Script to distribute inference tasks across multiple GPUs
# Usage: ./multi_gpu_inference.sh <checkpoints> <permutes> <num_gpus>

# Get command line arguments
CHECKPOINTS=$1
PERMUTES=$2
NUM_GPUS=${3:-8}  # Default to 8 GPUs if not specified

# Split the checkpoints into groups for each GPU
IFS=',' read -ra CKPT_ARRAY <<< "$CHECKPOINTS"
TOTAL_CKPTS=${#CKPT_ARRAY[@]}

# Calculate tasks per GPU
TASKS_PER_GPU=$(( (TOTAL_CKPTS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Distributing $TOTAL_CKPTS checkpoints across $NUM_GPUS GPUs ($TASKS_PER_GPU checkpoints per GPU)"

# Create output directory
OUTPUT_DIR="multi_gpu_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Launch processes for each GPU
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    # Calculate start and end indices for this GPU's checkpoints
    START_IDX=$((gpu * TASKS_PER_GPU))
    END_IDX=$(( (gpu + 1) * TASKS_PER_GPU - 1 ))
    
    # Ensure we don't go beyond the array bounds
    if [ $END_IDX -ge $TOTAL_CKPTS ]; then
        END_IDX=$((TOTAL_CKPTS - 1))
    fi
    
    # Skip if this GPU has no tasks
    if [ $START_IDX -gt $END_IDX ]; then
        continue
    fi
    
    # Build the checkpoint list for this GPU
    GPU_CKPTS=""
    for ((i=START_IDX; i<=END_IDX; i++)); do
        if [ -n "$GPU_CKPTS" ]; then
            GPU_CKPTS="$GPU_CKPTS,${CKPT_ARRAY[$i]}"
        else
            GPU_CKPTS="${CKPT_ARRAY[$i]}"
        fi
    done
    
    # Set the output file for this GPU
    OUTPUT_FILE="$OUTPUT_DIR/results_gpu${gpu}.json"
    
    echo "GPU $gpu: Processing checkpoints $START_IDX to $END_IDX"
    echo "GPU $gpu: Checkpoint list: $GPU_CKPTS"
    
    # Launch the process for this GPU
    CUDA_VISIBLE_DEVICES=$gpu python parallel_batch_inference.py \
        --checkpoints "$GPU_CKPTS" \
        --permutes $PERMUTES \
        --max_workers $PERMUTES \
        --output "$OUTPUT_FILE" > "$OUTPUT_DIR/log_gpu${gpu}.txt" 2>&1 &
    
    echo "Started process on GPU $gpu (PID $!)"
done

echo "All processes started. Logs and results will be saved in $OUTPUT_DIR/"
echo "Monitor progress with: tail -f $OUTPUT_DIR/log_gpu*.txt"
echo "Wait for all processes to complete with: wait"

# Wait for all background processes to complete
wait

echo "All GPU processes completed!"

# Combine results (optional)
echo "Results are available in $OUTPUT_DIR/"
