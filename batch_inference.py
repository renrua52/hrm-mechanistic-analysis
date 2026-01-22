import argparse
import yaml
import os
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt

from pretrain import PretrainConfig, init_train_state, create_dataloader
from eval_utils import load_checkpoint_and_config, forward_batch
from dataset.sudoku_transforms import sudoku_cyclic_shift

def main():
    parser = argparse.ArgumentParser(description="Batched Evaluation")
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--permutes", type=int, default=1)
    parser.add_argument("--num_batch", type=int, default=4646)
    parser.add_argument("--batch_size", type=int, default=91)
    args = parser.parse_args()
    ckpt_list = [p.strip() for p in args.checkpoints.split(",")]
    permutes = args.permutes
    num_batch = args.num_batch
    batch_size = args.batch_size

    torch.cuda.set_device(0)

    models = []
    
    for checkpoint_path in ckpt_list:
        checkpoint_file, config, checkpoint_dir = load_checkpoint_and_config(checkpoint_path)
        
        torch.random.manual_seed(config.seed)

        train_loader, train_metadata = create_dataloader(
            config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=1,  # Small batch for metadata only
            rank=0,
            world_size=1
        )
        
        # Initialize model (adapted from eval_checkpoint.py)
        train_state = init_train_state(config, train_metadata, world_size=1)

        # Load checkpoint weights (adapted from eval_checkpoint.py)
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cuda")
        try:
            train_state.model.load_state_dict(checkpoint, assign=True)
        except:
            # Handle torch.compile wrapped models
            cleaned_state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
            train_state.model.load_state_dict(cleaned_state_dict, assign=True)
        
        # Set model to evaluation mode
        train_state.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in train_state.model.parameters()):,}")

        models.append(train_state.model)
    
    # Load data once at the beginning
    print("Loading data...")
    all_inputs = torch.from_numpy(np.load("data/sudoku-extreme-1k-aug-1000/test/all__inputs.npy")).long().cuda()
    all_labels = torch.from_numpy(np.load("data/sudoku-extreme-1k-aug-1000/test/all__labels.npy")).long().cuda()
    print("Data loaded successfully!")
    
    # Pre-compute all permutations for each model
    correct = 0
    
    # Create a single tensor to track correctness for all examples
    all_correct = torch.zeros(num_batch * batch_size, dtype=bool, device="cuda")
    
    with torch.no_grad():
        for idx in range(num_batch):
            print(f'Batch {idx}/{num_batch}')
            
            # Get the current batch indices
            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            
            # Extract the current batch directly from pre-loaded data
            batch_inputs = all_inputs[start_idx:end_idx]
            batch_labels = all_labels[start_idx:end_idx]
            
            # Process each model
            for model in models:
                # Process all permutations at once if possible, otherwise in smaller chunks
                # Create a list to store predictions for all permutations
                all_batch_predictions = []
                all_batch_labels = []
                
                for perm in range(permutes):
                    # Apply permutation
                    inputs_perm = sudoku_cyclic_shift(batch_inputs, perm)
                    labels_perm = sudoku_cyclic_shift(batch_labels, perm)
                    
                    # Create batch dictionary
                    batch = {
                        "inputs": inputs_perm,
                        "labels": labels_perm,
                        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device="cuda")
                    }
                    
                    # Forward pass
                    results = forward_batch(model=model, batch=batch)
                    
                    # Store the final predictions
                    all_batch_predictions.append(results["all_predictions"][-1])
                    all_batch_labels.append(labels_perm)
                
                # Stack all predictions and check if any permutation is correct for each example
                stacked_predictions = torch.stack(all_batch_predictions, dim=0)  # [permutes, batch_size, 81]
                stacked_labels = torch.stack(all_batch_labels, dim=0)
                
                # Compare with labels
                equal_elements = (stacked_labels == stacked_predictions)  # [permutes, batch_size, 81]
                all_equal = torch.all(equal_elements, dim=2)  # [permutes, batch_size]
                any_perm_correct = torch.any(all_equal, dim=0)  # [batch_size]
                
                # Update the overall correctness
                all_correct[start_idx:end_idx] |= any_perm_correct
            
            # Clear unnecessary variables to free up memory
            del batch_inputs, batch_labels
            if 'batch' in locals():
                del batch
            if 'results' in locals():
                del results
            if 'stacked_predictions' in locals():
                del stacked_predictions
            if 'stacked_chunk' in locals():
                del stacked_chunk
            if 'equal_elements' in locals():
                del equal_elements
            if 'all_equal' in locals():
                del all_equal
            if 'any_perm_correct' in locals():
                del any_perm_correct
            
            # Explicitly clear the cache to free up memory
            torch.cuda.empty_cache()
        
        # Calculate the final correct count
        correct = all_correct.sum().item()
        
        # Print results
        total_examples = num_batch * batch_size
        print(f'{correct}/{total_examples}, accuracy={correct/total_examples}')
        
        print(f"\n{'='*60}")
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
