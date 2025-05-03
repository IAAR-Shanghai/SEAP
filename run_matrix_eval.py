import os
import subprocess
import itertools
from multiprocessing import Process
import time
import gc
import torch

# 你想执行的所有组合
prompt_types = ["cot", "icl", "knowledge", "zero_shot"]
calibration_tasks = [
    "gsm8k", "mathqa", "arc_easy", "arc_challenge",
    "openbookqa", "winogrande", "piqa", "hellaswag", "boolq"
    "wikitext2", "c4"
]
eval_tasks = [
    "gsm8k", "mathqa", "arc_easy", "arc_challenge",
    "openbookqa", "winogrande", "piqa", "hellaswag", "boolq"
]

# 基础参数模板
base_cmd = [
    "python", "evaluate_tasks.py",
    "--model_root_path", "/mnt/public/model/huggingface",
    "--model_name", "Llama-2-13b-hf",
    "--activations_root_path", "./activations",
    "--temp_dir", "./tmp",
    "--output_base_dir", "./eval_out"
]

# 所有组合
combos = list(itertools.product(prompt_types, calibration_tasks))

# 分成两组并行跑，每组绑定 2 个 GPU
device_groups = [["0,1"], ["2,3"]]


def run_job(prompt, calib, device_ids):
    cmd = base_cmd + [
        "--prompt_types", prompt,
        "--calibration_task", calib,
        "--task_types"
    ] + eval_tasks

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_ids

    print(f"[Spawn] prompt={prompt}, calib={calib}, GPUs={device_ids}")
    subprocess.run(cmd, env=env, check=True)


def main():
    for i in range(0, len(combos), 2):
        procs = []
        for j in range(2):
            if i + j >= len(combos):
                continue
            prompt, calib = combos[i + j]
            gpu_ids = device_groups[j % 2]
            p = Process(target=run_job, args=(prompt, calib, gpu_ids[0]))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)


if __name__ == "__main__":
    main()
