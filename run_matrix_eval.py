import os
import subprocess
import itertools
from multiprocessing import Process, Manager
import time
import torch
import argparse
import gc
import logging
import json
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matrix_eval.log')
    ]
)
logger = logging.getLogger(__name__)

# Task configurations
EXPERT_TASK_GROUPS = {
    "code_gen":       ["humaneval", "mbpp"],
    "math_reasoning": ["gsm8k", "mathqa"],
    "comparison":     ["boolq", "race"],
    "knowledge_qa":   ["arc_challenge", "arc_easy", "openbookqa"],
    "commonsense":    ["piqa", "winogrande", "hellaswag"],
}

LANGUAGE_CALIB_TASKS = ["wikitext2", "c4"]
ALL_TASKS = sorted(set(sum(EXPERT_TASK_GROUPS.values(), [])))

MODEL_NAMES = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
]

class ExperimentConfig:
    """Configuration class for matrix evaluation experiments."""
    def __init__(self, **kwargs):
        self.model_names: List[str] = kwargs.get('model_names', MODEL_NAMES)
        self.methods: List[str] = kwargs.get('methods', ["WIFV", "WIFN"])
        self.pruning_ratios: List[float] = kwargs.get('pruning_ratios', [0.2, 0.3, 0.5])
        self.expert_task_groups: Dict[str, List[str]] = kwargs.get('expert_task_groups', EXPERT_TASK_GROUPS)
        self.language_calib_tasks: List[str] = kwargs.get('language_calib_tasks', LANGUAGE_CALIB_TASKS)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_names': self.model_names,
            'methods': self.methods,
            'pruning_ratios': self.pruning_ratios,
            'expert_task_groups': self.expert_task_groups,
            'language_calib_tasks': self.language_calib_tasks
        }

def setup_experiment_directory(base_dir: str) -> str:
    """
    Create and setup experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        str: Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def run_job(cmd: List[str], device_ids: str, job_info: Dict[str, Any], 
           job_status: Dict[str, Any], job_id: int) -> None:
    """
    Run a single evaluation job.
    
    Args:
        cmd: Command to run
        device_ids: GPU device IDs to use
        job_info: Information about the job
        job_status: Shared dictionary to track job status
        job_id: Unique identifier for the job
    """
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_ids
        env["HF_ALLOW_CODE_EVAL"] = "1"

        job_desc = f"Job {job_id}: model={job_info['model_name']}, prompt={job_info['prompt']}, calib={job_info['calib']}"
        logger.info(f"Starting {job_desc} on GPUs {device_ids}")
        
        start_time = time.time()
        process = subprocess.run(cmd, env=env, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration = time.time() - start_time
        
        # Update job status
        job_status[job_id] = {
            "status": "completed",
            "duration": duration,
            "info": job_info,
            "return_code": process.returncode
        }
        
        logger.info(f"Completed {job_desc} in {duration:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error in {job_desc}: {str(e)}\nOutput: {e.output.decode() if e.output else 'None'}"
        logger.error(error_msg)
        job_status[job_id] = {
            "status": "failed",
            "error": error_msg,
            "info": job_info
        }
    except Exception as e:
        error_msg = f"Unexpected error in {job_desc}: {str(e)}"
        logger.error(error_msg)
        job_status[job_id] = {
            "status": "failed",
            "error": error_msg,
            "info": job_info
        }

def save_experiment_results(exp_dir: str, config: ExperimentConfig, 
                          job_status: Dict[str, Any]) -> None:
    """
    Save experiment configuration and results.
    
    Args:
        exp_dir: Experiment directory
        config: Experiment configuration
        job_status: Status of all jobs
    """
    results = {
        "config": config.to_dict(),
        "jobs": job_status,
        "summary": {
            "total_jobs": len(job_status),
            "completed": sum(1 for j in job_status.values() if j["status"] == "completed"),
            "failed": sum(1 for j in job_status.values() if j["status"] == "failed")
        }
    }
    
    results_file = os.path.join(exp_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

def main(args: argparse.Namespace) -> None:
    """
    Main function to run matrix evaluation experiments.
    
    Args:
        args: Command line arguments
    """
    try:
        # Setup experiment directory
        exp_dir = setup_experiment_directory(args.output_base_dir)
        logger.info(f"Experiment directory: {exp_dir}")
        
        # Setup GPU allocation
        gpus = list(map(str, range(torch.cuda.device_count())))
        if not gpus:
            raise RuntimeError("No GPU devices found")
            
        gpus_per_thread = max(1, len(gpus) // args.num_threads)
        device_groups = [
            ",".join(gpus[i * gpus_per_thread: (i + 1) * gpus_per_thread])
            for i in range(args.num_threads)
        ]
        logger.info(f"Using {args.num_threads} threads with device groups: {device_groups}")

        # Base command
        base_cmd = [
            "python", "evaluate_tasks.py",
            "--model_root_path", args.model_root_path,
            "--activations_root_path", args.activations_root_path,
            "--temp_dir", args.temp_dir,
            "--output_base_dir", exp_dir,
        ]

        # Generate jobs
        config = ExperimentConfig()
        jobs = []

        for model_name in config.model_names:
            for method in config.methods:
                for pruning_ratio in config.pruning_ratios:
                    # Expert tasks
                    for expert, tasks in config.expert_task_groups.items():
                        jobs.append({
                            "model_name": model_name,
                            "prompt": "experts",
                            "calib": expert,
                            "tasks": tasks,
                            "method": method,
                            "ratio": pruning_ratio,
                            "strategy": "retention"
                        })

                    # Language tasks
                    for calib in config.language_calib_tasks:
                        jobs.append({
                            "model_name": model_name,
                            "prompt": "zero_shot",
                            "calib": calib,
                            "tasks": ALL_TASKS,
                            "method": method,
                            "ratio": pruning_ratio,
                            "strategy": "uniform"
                        })

        logger.info(f"Generated {len(jobs)} total jobs")

        # Setup job status tracking
        manager = Manager()
        job_status = manager.dict()

        # Run jobs in batches
        for i in range(0, len(jobs), args.num_threads):
            procs = []
            batch_start = time.time()
            
            for j in range(args.num_threads):
                if i + j >= len(jobs):
                    break
                    
                job = jobs[i + j]
                gpu_ids = device_groups[j]
                job_id = i + j

                cmd = base_cmd + [
                    "--model_name", job["model_name"],
                    "--prompt_types", job["prompt"],
                    "--calibration_task", job["calib"],
                    "--method", job["method"],
                    "--sparsity_strategy", job["strategy"],
                    "--pruning_ratio", str(job["ratio"]),
                    "--task_types"
                ] + job["tasks"]

                p = Process(target=run_job, 
                          args=(cmd, gpu_ids, job, job_status, job_id))
                p.start()
                procs.append(p)

            # Wait for batch completion
            for p in procs:
                p.join()
                
            batch_duration = time.time() - batch_start
            logger.info(f"Completed batch {i//args.num_threads + 1} in {batch_duration:.2f} seconds")
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        # Save results
        save_experiment_results(exp_dir, config, dict(job_status))
        logger.info("Matrix evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error in matrix evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matrix evaluation experiments")
    
    parser.add_argument("--num_threads", type=int, default=4,
                      help="Number of parallel threads to use")
    parser.add_argument("--model_root_path", type=str, 
                      default="/mnt/public/model/huggingface",
                      help="Root directory containing models")
    parser.add_argument("--activations_root_path", type=str, 
                      default="./activations",
                      help="Directory containing activation data")
    parser.add_argument("--temp_dir", type=str, 
                      default="./tmp",
                      help="Directory for temporary files")
    parser.add_argument("--output_base_dir", type=str, 
                      default="./eval_out",
                      help="Base directory for evaluation outputs")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    main(args)
