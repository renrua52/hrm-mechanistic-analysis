import argparse
import yaml
import os
import time
import json
from typing import Dict, Any, List
import concurrent.futures
import threading
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import necessary components from existing files
from pretrain import PretrainConfig, init_train_state, create_dataloader
from eval_utils import load_checkpoint_and_config, create_batch, forward_batch

# Global lock for thread synchronization when updating shared resources
result_lock = threading.Lock()
gpu_memory_lock = threading.Lock()

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        with gpu_memory_lock:
            # Get the current device
            device = torch.cuda.current_device()
            
            # Get memory information
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - reserved_memory
            
            return {
                "total": total_memory,
                "reserved": reserved_memory,
                "allocated": allocated_memory,
                "free": total_memory - reserved_memory
            }
    return None

def process_batch(model, train_loader, train_metadata, puzzle_id, idx, batch_size, permute, model_idx, perm_idx, total_tasks):
    """Process a single batch with a specific model and permutation"""
    try:
        # Log task start
        task_id = f"Model {model_idx+1}, Permutation {perm_idx+1}"
        
        # Create batch
        batch = create_batch(
            train_loader=train_loader,
            train_metadata=train_metadata,
            puzzle_id=puzzle_id,
            idx=idx,
            batch_size=batch_size,
            permute=permute
        )
        
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()  # (B, 81)
        
        # Forward pass
        results = forward_batch(
            model=model,
            batch=batch,
        )
        
        # Calculate correctness
        equal_elements = (labels == results["all_predictions"][-1].cpu().numpy())  # Shape: (batch_size, 81)
        correct_samples = np.all(equal_elements, axis=1)  # Shape: (batch_size,)
        
        # Clean up to free memory
        del batch, results, equal_elements
        torch.cuda.empty_cache()
        
        return correct_samples
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        # Return empty result in case of error
        return np.zeros(batch_size, dtype=bool)

def main():
    parser = argparse.ArgumentParser(description="Parallel Batched Evaluation")
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--permutes", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None, 
                        help="Maximum number of worker threads (default: number of checkpoints * permutes)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (default: None)")
    parser.add_argument("--num_batch", type=int, default=182,
                        help="Number of batches to process (default: 182)")
    parser.add_argument("--batch_size", type=int, default=2323,
                        help="Batch size for each inference (default: 2323)")
    parser.add_argument("--gpu_memory_limit", type=float, default=0.9,
                        help="Fraction of GPU memory to use (default: 0.9)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    ckpt_list = [p.strip() for p in args.checkpoints.split(",")]
    permutes = args.permutes
    
    # Set max_workers to the number of tasks if not specified
    if args.max_workers is None:
        max_workers = len(ckpt_list) * permutes
    else:
        max_workers = min(args.max_workers, len(ckpt_list) * permutes)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using CUDA device 0")
    else:
        print("CUDA not available, using CPU")
    
    # Load all models
    models = []
    train_loaders = []
    train_metadatas = []
    
    print(f"Loading {len(ckpt_list)} checkpoints...")
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
        
        # Initialize model
        train_state = init_train_state(config, train_metadata, world_size=1)
        
        # Load checkpoint weights
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
        train_loaders.append(train_loader)
        train_metadatas.append(train_metadata)
    
    num_batch = args.num_batch
    batch_size = args.batch_size
    correct = 0
    
    # Calculate total number of tasks
    total_tasks = len(models) * permutes
    
    print(f"Starting parallel inference with {max_workers} workers...")
    print(f"Processing {len(models)} models with {permutes} permutations each ({total_tasks} total tasks)")
    
    # Create results dictionary
    results = {
        "config": {
            "checkpoints": ckpt_list,
            "permutes": permutes,
            "max_workers": max_workers,
            "num_batch": num_batch,
            "batch_size": batch_size,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "batch_results": [],
        "overall_results": {}
    }
    
    with torch.no_grad():
        # Create progress bar for batches
        batch_pbar = tqdm(total=num_batch, desc="Batches", position=0)
        
        for idx in range(num_batch):
            batch_start_time = time.time()
            
            B = batch_size
            all_correct = np.zeros(B, dtype=bool)
            
            # Create a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for each model and permutation
                future_to_task = {}
                for model_idx, model in enumerate(models):
                    for perm_idx in range(permutes):
                        future = executor.submit(
                            process_batch,
                            model,
                            train_loaders[model_idx],
                            train_metadatas[model_idx],
                            0,  # puzzle_id
                            idx,
                            batch_size,
                            perm_idx,
                            model_idx,
                            perm_idx,
                            total_tasks
                        )
                        future_to_task[(model_idx, perm_idx)] = future
                
                # Create progress bar for tasks within this batch
                task_pbar = tqdm(total=len(future_to_task), desc=f"Tasks (Batch {idx+1}/{num_batch})", position=1, leave=False)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_task.values()):
                    try:
                        correct_samples = future.result()
                        with result_lock:
                            all_correct = all_correct | correct_samples
                        task_pbar.update(1)
                    except Exception as exc:
                        print(f'Task generated an exception: {exc}')
                        task_pbar.update(1)
                
                task_pbar.close()
            
            batch_correct = all_correct.sum()
            correct += batch_correct
            batch_accuracy = batch_correct / batch_size
            
            # Record batch results
            batch_result = {
                "batch_idx": idx,
                "correct": int(batch_correct),
                "total": batch_size,
                "accuracy": float(batch_accuracy),
                "time_seconds": time.time() - batch_start_time
            }
            results["batch_results"].append(batch_result)
            
            # Update progress
            batch_pbar.set_postfix({
                "correct": f"{correct}/{(idx+1)*batch_size}",
                "accuracy": f"{correct/((idx+1)*batch_size):.4f}"
            })
            batch_pbar.update(1)
            
            # Free memory
            torch.cuda.empty_cache()
            
            # Save intermediate results if output path is specified
            if args.output:
                with open(args.output, 'w') as f:
                    # Update overall results
                    results["overall_results"] = {
                        "correct": int(correct),
                        "total": (idx+1) * batch_size,
                        "accuracy": float(correct / ((idx+1) * batch_size)),
                        "elapsed_time": time.time() - start_time
                    }
                    json.dump(results, f, indent=2)
        
        batch_pbar.close()
        
        # Calculate final results
        final_accuracy = correct / (num_batch * batch_size)
        elapsed_time = time.time() - start_time
        
        # Update overall results
        results["overall_results"] = {
            "correct": int(correct),
            "total": num_batch * batch_size,
            "accuracy": float(final_accuracy),
            "elapsed_time": elapsed_time,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save final results if output path is specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Print final results
        print(f"\n{'='*60}")
        print(f"RESULTS: {correct}/{num_batch*batch_size}, accuracy={final_accuracy:.6f}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"{'='*60}")
        print("PARALLEL INFERENCE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    exit(main())
