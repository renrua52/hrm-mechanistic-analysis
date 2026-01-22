#!/usr/bin/env python3
"""
Script to combine results from multiple GPU runs into a single result file.
This script takes the individual JSON result files from each GPU and combines them,
calculating the overall accuracy across all GPUs.
"""

import os
import json
import argparse
import glob
from datetime import datetime

def combine_results(results_dir, output_file=None):
    """
    Combine results from multiple JSON files in the given directory.
    
    Args:
        results_dir: Directory containing the JSON result files
        output_file: Path to save the combined results (default: results_dir/combined_results.json)
    
    Returns:
        Combined results dictionary
    """
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(results_dir, "results_gpu*.json"))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    # Initialize combined results
    combined_results = {
        "config": {
            "combined_from": len(json_files),
            "source_files": json_files,
            "combination_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "batch_results": [],
        "overall_results": {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "elapsed_time": 0.0
        }
    }
    
    # Load and combine results
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                
                # Add config info from first file if not already present
                if "checkpoints" not in combined_results["config"] and "checkpoints" in data["config"]:
                    combined_results["config"]["checkpoints"] = data["config"]["checkpoints"]
                    combined_results["config"]["permutes"] = data["config"]["permutes"]
                    combined_results["config"]["num_batch"] = data["config"]["num_batch"]
                    combined_results["config"]["batch_size"] = data["config"]["batch_size"]
                    combined_results["config"]["start_time"] = data["config"]["start_time"]
                
                # Extend batch results
                combined_results["batch_results"].extend(data["batch_results"])
                
                # Add to overall results
                combined_results["overall_results"]["correct"] += data["overall_results"]["correct"]
                combined_results["overall_results"]["total"] += data["overall_results"]["total"]
                combined_results["overall_results"]["elapsed_time"] = max(
                    combined_results["overall_results"]["elapsed_time"],
                    data["overall_results"]["elapsed_time"]
                )
                
                print(f"Processed {json_file}: {data['overall_results']['correct']}/{data['overall_results']['total']} correct")
                
            except json.JSONDecodeError:
                print(f"Error: Could not parse {json_file} as JSON")
            except KeyError as e:
                print(f"Error: Missing key {e} in {json_file}")
    
    # Calculate overall accuracy
    if combined_results["overall_results"]["total"] > 0:
        combined_results["overall_results"]["accuracy"] = (
            combined_results["overall_results"]["correct"] / 
            combined_results["overall_results"]["total"]
        )
    
    # Sort batch results by batch_idx
    combined_results["batch_results"].sort(key=lambda x: x["batch_idx"])
    
    # Add end time
    combined_results["overall_results"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save combined results
    if output_file is None:
        output_file = os.path.join(results_dir, "combined_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results:")
    print(f"Correct: {combined_results['overall_results']['correct']}")
    print(f"Total: {combined_results['overall_results']['total']}")
    print(f"Accuracy: {combined_results['overall_results']['accuracy']:.6f}")
    print(f"Max elapsed time: {combined_results['overall_results']['elapsed_time']:.2f} seconds")
    print(f"\nResults saved to {output_file}")
    
    return combined_results

def main():
    parser = argparse.ArgumentParser(description="Combine results from multiple GPU runs")
    parser.add_argument("results_dir", help="Directory containing the JSON result files")
    parser.add_argument("--output", "-o", help="Path to save the combined results")
    
    args = parser.parse_args()
    
    combine_results(args.results_dir, args.output)

if __name__ == "__main__":
    main()
