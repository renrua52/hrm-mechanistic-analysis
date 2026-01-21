import argparse
import yaml
import os
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import necessary components from existing files
from pretrain import PretrainConfig, init_train_state, create_dataloader
from eval_utils import load_checkpoint_and_config, create_batch


def get_correct(batch: Dict[str, torch.Tensor], results: List[Dict[str, Any]]):
    inputs = batch["inputs"].cpu().numpy()
    labels = batch["labels"].cpu().numpy() # (B, 81)
    B = labels.shape[0]
    all_correct = np.zeros(B, dtype=bool)
    for result in results:
        equal_elements = (labels == result["all_predictions"][15].cpu().numpy())  # 形状: (batch_size, 81)
        all_correct = all_correct | np.all(equal_elements, axis=1)  # 形状: (batch_size,)
    
    correct = all_correct.sum()

    return correct


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Using CUDA device 0")
    else:
        print("CUDA not available, using CPU")

    # ckpt_list = ["/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 vermilion-cheetah/step_26040"]
    # ckpt_list = ["/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000-hint-rand-all ACT-torch/HierarchicalReasoningModel_ACTV1 burrowing-pudu/step_52080"]

    ckpt_list = [f"/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000-hint-rand-all ACT-torch/HierarchicalReasoningModel_ACTV1 burrowing-pudu/step_{i*2604}" for i in range(15, 25)]

    models = []
    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=1,  # Small batch for metadata only
        rank=0,
        world_size=1
    )
    
    for checkpoint_path in ckpt_list:
        checkpoint_file, config, checkpoint_dir = load_checkpoint_and_config(checkpoint_path)
        
        torch.random.manual_seed(config.seed)
        
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
    
    num_batch = 1000
    correct = 0
    batch_size = 100

    with torch.no_grad():
        for idx in range(num_batch):
            print(f'-----Testing batch {idx}/{num_batch} -----')
            # Create hand-coded example
            

            # Forward the example
            B = batch_size
            all_correct = np.zeros(B, dtype=bool)
            for model in models:
                for perm in range(9):
                    batch = create_batch(
                        train_loader=train_loader,
                        train_metadata=train_metadata,
                        puzzle_id=0,
                        idx=idx,
                        batch_size=batch_size,
                        permute=perm
                    )
                    inputs = batch["inputs"].cpu().numpy()
                    labels = batch["labels"].cpu().numpy() # (B, 81)
                    results = forward_batch(
                        model=model,
                        batch=batch,
                        max_steps=1000
                    )
                    equal_elements = (labels == results["all_predictions"][15].cpu().numpy())  # 形状: (batch_size, 81)
                    all_correct = all_correct | np.all(equal_elements, axis=1)  # 形状: (batch_size,)

            correct += all_correct.sum()

            del resultss, batch
            torch.cuda.empty_cache()
        
        # Print results
        print(f'{correct}/{num_batch*batch_size}')
        
        print(f"\n{'='*60}")
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())