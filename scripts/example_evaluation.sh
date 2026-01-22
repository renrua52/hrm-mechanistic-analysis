CKPTS=$(python -c "
prefix='checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/HierarchicalReasoningModel_ACTV1 military-salamander/step_'
print(','.join([f'{prefix}{i*1302}' for i in range(31, 41)]))
")
python batch_inference.py --checkpoints "$CKPTS" --permute 9
# For a snapshot result, use the following instead:
# python batch_inference.py --checkpoints "$CKPTS" --num_batch 10 --batch_size 100 --permute 9