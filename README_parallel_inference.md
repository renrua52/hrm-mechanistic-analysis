# Parallel Batch Inference

This script provides a parallelized version of the batch inference process for evaluating multiple checkpoints and permutations simultaneously. It significantly improves performance by running multiple inference tasks in parallel using threading.

## Features

- **Parallel Processing**: Run multiple model checkpoints and permutations simultaneously
- **Progress Tracking**: Visual progress bars for both batches and individual tasks
- **Result Aggregation**: Combines results using logical OR operation as in the original script
- **Memory Management**: Efficient GPU memory handling to prevent out-of-memory errors
- **Result Saving**: Option to save results to a JSON file for later analysis
- **Customizable Parameters**: Flexible command-line arguments for various settings

## Usage

```bash
python parallel_batch_inference.py --checkpoints "checkpoint1,checkpoint2,checkpoint3" --permutes 5 [options]
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoints` | Comma-separated list of checkpoint paths | (Required) |
| `--permutes` | Number of permutations to run for each checkpoint | 1 |
| `--max_workers` | Maximum number of worker threads | Number of tasks (checkpoints × permutes) |
| `--output` | Path to save results as JSON | None (no saving) |
| `--num_batch` | Number of batches to process | 182 |
| `--batch_size` | Batch size for each inference | 2323 |
| `--gpu_memory_limit` | Fraction of GPU memory to use | 0.9 |

## Example

```bash
# Run with 3 checkpoints, 5 permutations each, and save results
python parallel_batch_inference.py \
  --checkpoints "checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint,checkpoint2,checkpoint3" \
  --permutes 5 \
  --max_workers 8 \
  --output results.json
```

## Performance Considerations

- **Number of Workers**: The optimal number of worker threads depends on your GPU memory and the size of your models. If you encounter out-of-memory errors, try reducing the `--max_workers` value.
- **Batch Size**: You can adjust the batch size with `--batch_size` to balance between memory usage and throughput.
- **GPU Memory**: The script monitors GPU memory usage and can be configured with `--gpu_memory_limit` to prevent using too much GPU memory.

## Output Format

When using the `--output` option, the script saves results in JSON format with the following structure:

```json
{
  "config": {
    "checkpoints": ["checkpoint1", "checkpoint2", ...],
    "permutes": 5,
    "max_workers": 8,
    "num_batch": 182,
    "batch_size": 2323,
    "start_time": "2026-01-22 15:55:00"
  },
  "batch_results": [
    {
      "batch_idx": 0,
      "correct": 1500,
      "total": 2323,
      "accuracy": 0.6457,
      "time_seconds": 45.2
    },
    ...
  ],
  "overall_results": {
    "correct": 350000,
    "total": 422786,
    "accuracy": 0.8278,
    "elapsed_time": 3600.5,
    "end_time": "2026-01-22 16:55:00"
  }
}
```

## Comparison with Original Implementation

The original `batch_inference.py` script processes each checkpoint and permutation sequentially, which can be very time-consuming when dealing with multiple checkpoints and permutations. This parallelized version significantly reduces the total execution time by running multiple tasks simultaneously.

For example, with 3 checkpoints and 5 permutations each (15 total tasks), the original script would run these 15 tasks one after another. The parallelized version can run all 15 tasks simultaneously (or a configurable number of them based on available resources), potentially reducing the execution time by up to 15x in this example.

## Error Handling

The script includes robust error handling to ensure that a failure in one task doesn't cause the entire process to fail. If a task encounters an error, it will be logged, and the script will continue with the remaining tasks.
