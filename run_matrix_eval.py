import os
import subprocess
import itertools
from multiprocessing import Process
import time
import gc
import torch
import argparse


def run_job(prompt, calib, device_ids, base_cmd, eval_tasks):
    cmd = base_cmd + [
        "--prompt_types", prompt,
        "--calibration_task", calib,
        "--task_types"
    ] + eval_tasks

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_ids

    print(f"[Spawn] prompt={prompt}, calib={calib}, GPUs={device_ids}")
    subprocess.run(cmd, env=env, check=True)


def main(args):
    prompt_types = ["cot", "icl", "knowledge", "zero_shot"]
    calibration_tasks = [
        "gsm8k", "mathqa", "arc_easy", "arc_challenge",
        "openbookqa", "winogrande", "piqa", "hellaswag", "boolq",
        "wikitext2", "c4"
    ]
    eval_tasks = [
        "gsm8k", "mathqa", "arc_easy", "arc_challenge",
        "openbookqa", "winogrande", "piqa", "hellaswag", "boolq"
    ]

    base_cmd = [
        "python", "evaluate_tasks.py",
        "--model_root_path", "/mnt/public/model/huggingface",
        "--model_name", "Llama-2-13b-hf",
        "--activations_root_path", "./activations",
        "--temp_dir", "./tmp",
        "--output_base_dir", "./eval_out"
    ]

    combos = list(itertools.product(prompt_types, calibration_tasks))

    # e.g. for 4 GPUs and 2 threads => [ "0,1", "2,3" ]
    gpus = list(map(str, range(torch.cuda.device_count())))
    gpus_per_thread = len(gpus) // args.num_threads
    device_groups = [
        ",".join(gpus[i * gpus_per_thread: (i + 1) * gpus_per_thread])
        for i in range(args.num_threads)
    ]

    print(f"🔧 Using {args.num_threads} threads with device groups: {device_groups}")

    for i in range(0, len(combos), args.num_threads):
        procs = []
        for j in range(args.num_threads):
            if i + j >= len(combos):
                continue
            prompt, calib = combos[i + j]
            gpu_ids = device_groups[j]
            p = Process(target=run_job, args=(prompt, calib, gpu_ids, base_cmd, eval_tasks))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of parallel threads (each uses N GPUs)")
    args = parser.parse_args()
    main(args)
