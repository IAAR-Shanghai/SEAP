import os
import subprocess
import itertools
from multiprocessing import Process
import time
import gc
import torch
import argparse


def run_job(prompt, calib, device_ids, base_cmd, eval_tasks, model_name):
    cmd = base_cmd + [
        "--model_name", model_name,
        "--prompt_types", prompt,
        "--calibration_task", calib,
        "--task_types"
    ] + eval_tasks

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_ids
    env["HF_ALLOW_CODE_EVAL"] = "1"  # 🛡️ 对 mbpp/humaneval 自动放行

    print(f"[Spawn] model={model_name}, prompt={prompt}, calib={calib}, GPUs={device_ids}")
    subprocess.run(cmd, env=env, check=True)


def main(args):
    # 默认配置
    prompt_types = args.prompt_types or ["zero_shot", "cot", "icl", "knowledge", "corpus"]
    calibration_tasks = args.calibration_tasks or [
        "mbpp", "humaneval",
        "gsm8k", "mathqa",
        "boolq", "race",
        "arc_easy", "arc_challenge", "openbookqa", 
        "winogrande", "piqa", "hellaswag", 
        "wikitext2", "c4"
    ]
    eval_tasks = args.eval_tasks or [
        "mbpp", "humaneval",
        "gsm8k", "mathqa",
        "boolq", "race",
        "arc_easy", "arc_challenge", "openbookqa", 
        "winogrande", "piqa", "hellaswag", 
    ]
    model_names = args.model_names or ["Llama-2-7b-hf", "Llama-2-13b-hf"]

    base_cmd = [
        "python", "evaluate_tasks.py",
        "--model_root_path", args.model_root_path,
        "--activations_root_path", args.activations_root_path,
        "--temp_dir", args.temp_dir,
        "--output_base_dir", args.output_base_dir,
    ]

    # GPU 分配
    gpus = list(map(str, range(torch.cuda.device_count())))
    gpus_per_thread = len(gpus) // args.num_threads
    device_groups = [
        ",".join(gpus[i * gpus_per_thread: (i + 1) * gpus_per_thread])
        for i in range(args.num_threads)
    ]

    print(f"🔧 Using {args.num_threads} threads with device groups: {device_groups}")

    # 每个模型单独跑一轮
    for model_name in model_names:
        print(f"\n🚀 Running evaluation for model: {model_name}")
        combos = list(itertools.product(prompt_types, calibration_tasks))

        for i in range(0, len(combos), args.num_threads):
            procs = []
            for j in range(args.num_threads):
                if i + j >= len(combos):
                    continue
                prompt, calib = combos[i + j]
                gpu_ids = device_groups[j]
                p = Process(target=run_job, args=(prompt, calib, gpu_ids, base_cmd, eval_tasks, model_name))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=4, help="Number of parallel threads")
    parser.add_argument("--model_names", nargs="+", default=None)
    parser.add_argument("--prompt_types", nargs="+", default=None)
    parser.add_argument("--calibration_tasks", nargs="+", default=None)
    parser.add_argument("--eval_tasks", nargs="+", default=None)

    parser.add_argument("--model_root_path", type=str, default="/mnt/public/model/huggingface")
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--temp_dir", type=str, default="./tmp")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out")

    args = parser.parse_args()
    main(args)
