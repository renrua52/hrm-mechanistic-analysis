python batch_inference.py --checkpoints "checkpoints/Maze/maze/checkpoint" \
--dataset maze \
--num_batch 10 --batch_size 100 --permutes 2

# Do not set --permute value other than 1 or 2 for maze ckpt evaluation.
# You may alternatively download the trained ckpt from the official HRM repo (https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard).