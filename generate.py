import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

def main(args):
    device = torch.device('cuda')

    # 尝试使用 AutoModelForCausalLM 加载模型
    try:
        print(f"Loading model from {args.model_root_path}/{args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(args.model_root_path, args.model_name), 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map="auto"  # Use device_map to automatically place model on multiple GPUs if available
        )
        print("Model loaded successfully using AutoModelForCausalLM.from_pretrained")
    except Exception as e:
        print(f"Failed to load model using AutoModelForCausalLM: {e}")
        print(f"Loading model from {args.model_root_path}/{args.model_name}/pruned_model.pt...")
        model = torch.load(os.path.join(args.model_root_path, args.model_name, "pruned_model.pt"), map_location=device)
        print("Model loaded successfully using torch.load")
    
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_root_path, args.model_name))

    generate_kwargs = {
        "max_new_tokens": 300,
        "min_new_tokens": 200,
        "temperature": 0.1,
        "do_sample": False, 
        "top_k": 3,
        "penalty_alpha": 0.6,
    }

    prompts = [
        "AI can create a logo in seconds.",
        "What is McDonald's?",
    ]

    # 输出路径，创建包含模型名称的子文件夹
    output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_path, exist_ok=True)

    total_tokens_generated = 0  # 用于计算总生成的token数量
    generated_results = []  # 用来存储所有生成的结果

    # 先处理所有输入和生成 ID
    input_ids_list = []
    for prompt in prompts:
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            assert len(input_ids) == 1, len(input_ids)
            if input_ids[0][-1] == 2:  # EOS token handling
                input_ids = input_ids[:, :-1]
            input_ids_list.append(input_ids.to(device))
   
    # 记录生成开始时间
    start_time = time.time()

    # 开始生成
    for input_ids in input_ids_list:
        # 生成文本
        generated_ids = model.generate(input_ids, **generate_kwargs)
        result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        
        # 将生成结果保存到列表
        generated_results.append(result[0])
    
    end_time = time.time()

    # 计算总生成的 tokens 数量
    for result in generated_results:
        total_tokens_generated += len(tokenizer.encode(result))

    # 计算每秒生成的 token 数量
    elapsed_time = end_time - start_time
    tokens_per_sec = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0

    # 打印所有生成的结果以及相关统计信息
    for idx, result in enumerate(generated_results):
        print(f"Generated (Prompt {idx + 1}): {result}")
    
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} Tokens/s")

    # 保存生成的结果到文件
    for idx, result in enumerate(generated_results):
        output_file_path = os.path.join(output_path, f"output_{prompts[idx][:10]}.txt")
        with open(output_file_path, 'w') as f:
            f.write(f"Prompt: {prompts[idx]}\n")
            f.write(f"Generated: {result}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pretrained language model.")

    parser.add_argument("--model_root_path", type=str, required=True, help="Root directory where the model is stored.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or directory under --model_root_path.")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Directory to save generated text outputs.")
    
    args = parser.parse_args()
    main(args)
