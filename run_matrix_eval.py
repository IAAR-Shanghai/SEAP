import os
import subprocess
import itertools
from multiprocessing import Process
import time
import torch
import argparse
import gc


EXPERT_TASK_GROUPS = {
    "code_gen":       ["humaneval", "mbpp"],
    "math_reasoning": ["gsm8k", "mathqa"],
    "comparison":     ["boolq", "race"],
    "knowledge_qa":   ["arc_challenge", "arc_easy", "openbookqa"],
    "commonsense":    ["piqa", "winogrande", "hellaswag"],
}

LANGUAGE_CALIB_TASKS = ["wikitext2", "c4"]
ALL_TASKS = sorted(set(sum(EXPERT_TASK_GROUPS.values(), [])))  # 去重合并

# ✅ 内置模型名称
MODEL_NAMES = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
]


def run_job(cmd, device_ids, model_name, prompt, calib):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_ids
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"[Spawn] model={model_name}, prompt={prompt}, calib={calib}, GPUs={device_ids}")
    subprocess.run(cmd, env=env, check=True)


def main(args):
    gpus = list(map(str, range(torch.cuda.device_count())))
    gpus_per_thread = len(gpus) // args.num_threads
    device_groups = [
        ",".join(gpus[i * gpus_per_thread: (i + 1) * gpus_per_thread])
        for i in range(args.num_threads)
    ]

    base_cmd = [
        "python", "evaluate_tasks.py",
        "--model_root_path", args.model_root_path,
        "--activations_root_path", args.activations_root_path,
        "--temp_dir", args.temp_dir,
        "--output_base_dir", args.output_base_dir,
    ]

    print(f"🔧 Using {args.num_threads} threads with device groups: {device_groups}")

    jobs = []

    for model_name in MODEL_NAMES:
        for method in ["WIFV", "WIFN"]:
            for pruning_ratio in [0.2, 0.3, 0.5]:
                # 🧠 Experts
                for expert, tasks in EXPERT_TASK_GROUPS.items():
                    jobs.append({
                        "model_name": model_name,
                        "prompt": "experts",
                        "calib": expert,
                        "tasks": tasks,
                        "method": method,
                        "ratio": pruning_ratio,
                        "strategy": "retention"
                    })

                # 💬 Language experts
                for calib in LANGUAGE_CALIB_TASKS:
                    jobs.append({
                        "model_name": model_name,
                        "prompt": "zero_shot",
                        "calib": calib,
                        "tasks": ALL_TASKS,
                        "method": method,
                        "ratio": pruning_ratio,
                        "strategy": "uniform"
                    })

    print(f"\n📋 Total jobs: {len(jobs)}")

    for i in range(0, len(jobs), args.num_threads):
        procs = []
        for j in range(args.num_threads):
            if i + j >= len(jobs):
                continue
            job = jobs[i + j]
            gpu_ids = device_groups[j]

            cmd = base_cmd + [
                "--model_name", job["model_name"],
                "--prompt_types", job["prompt"],
                "--calibration_task", job["calib"],
                "--method", job["method"],
                "--sparsity_strategy", job["strategy"],
                "--pruning_ratio", str(job["ratio"]),
                "--task_types"
            ] + job["tasks"]

            p = Process(target=run_job, args=(cmd, gpu_ids, job["model_name"], job["prompt"], job["calib"]))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    print("✅ All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--model_root_path", type=str, default="/mnt/public/model/huggingface")
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--temp_dir", type=str, default="./tmp")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out")
    args = parser.parse_args()
    main(args)
