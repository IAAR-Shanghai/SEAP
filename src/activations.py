# activations.py
import os
import gc
import random
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class OnlineStats:
    """
    使用在线更新策略来维护以下统计量：
      - running_mean: 每个特征维度的平均值
      - running_var:  每个特征维度的波动量（近似方差）
      - running_l2:   每个特征维度的 L2 均值 (平均 x^2)

    思路：在接收到新的批次 x 时，将已有统计量按一定比例缩放后，再与该批次的统计量按相反比例融合。
         这样我们无需存储所有样本数据，就能动态估计平均值、近似方差以及 L2 均值。
    """

    def __init__(self, hidden_dim=None):
        self.hidden_dim = hidden_dim

        self.running_mean = None  # [D], 当前累积的特征平均值
        self.running_var  = None  # [D], 当前累积的波动量(近似方差)
        self.running_l2   = None  # [D], 平均 x^2，用于衡量特征分量的能量
        self.sample_count = 0     # 当前所统计的总样本数(包括batch与sequence维度)

    def update(self, x: torch.Tensor):
        """
        用一个新批次 x 来更新统计量。
        支持形状 (batch, length, dim) 或 (N, dim)；在内部先 flatten 为 (N, D)。

        步骤：
          1. flatten 为 [N, D]
          2. clamp, 避免数值过大或过小
          3. 若是首次更新，则直接用该批次初始化统计量
          4. 若已有统计量，则按在线合并方式更新 mean, var, l2
        """
        # 1. flatten => [N, D]
        x = x.view(-1, x.shape[-1])

        # 2. clamp，避免出现极端大数导致浮点溢出
        x = x.to(torch.float32)  # 或 x.float()

        # debug: 检查 Inf/NaN
        if torch.isinf(x).any():
            print(f"[OnlineStats] After clamp, x still has Inf, shape={x.shape}")
        if torch.isnan(x).any():
            print(f"[OnlineStats] x has NaN after clamp, shape={x.shape}")

        n_new = x.shape[0]
        
        # 3. 如果是第一次更新，则直接初始化
        if self.sample_count == 0:
            self.hidden_dim = x.shape[-1]
            self.running_mean = x.mean(dim=0)        # [D]
            self.running_var  = torch.zeros_like(self.running_mean)
            self.running_l2   = (x ** 2).mean(dim=0) # [D]
            self.sample_count = n_new
            return

        # 4. 若已有统计量，则在线合并
        old_mean = self.running_mean.clone()
        total_count = self.sample_count + n_new

        alpha_old = self.sample_count / total_count
        alpha_new = n_new / total_count

        batch_mean = x.mean(dim=0)  # [D]
        self.running_mean = alpha_old * self.running_mean + alpha_new * batch_mean

        # 4.1 更新波动量 running_var
        if total_count > 1:
            self.running_var *= (self.sample_count - 1) / (total_count - 1)

        diff_sum = ((x - self.running_mean) * (x - old_mean)).sum(dim=0)  # [D]

        self.running_var += diff_sum / total_count

        # 4.2 更新 L2 均值
        batch_l2 = (x ** 2).mean(dim=0)  # [D]
        self.running_l2 = alpha_old * self.running_l2 + alpha_new * batch_l2

        # 4.3 更新计数
        self.sample_count = total_count

    def get_stats(self):
        """
        返回当前的统计量: mean, var, l2。
        """
        if self.sample_count == 0:
            return {
                "mean": None,
                "var":  None,
                "l2":   None
            }
        return {
            "mean": self.running_mean,
            "var":  self.running_var,
            "l2":   self.running_l2
        }

    def reset(self):
        """
        重置统计结果，清空已记录的数据。
        """
        self.running_mean = None
        self.running_var  = None
        self.running_l2   = None
        self.sample_count = 0

class ActivationHookManager:
    """
    管理前向传播 Hook，用于收集激活值。
    """
    def __init__(self):
        self.layer_activations = {}

    def _init_stats_dict(self):
        return {
            'attention_input_states': OnlineStats(),
            'attention_post_aggregation': OnlineStats(),
            'mlp_input_states': OnlineStats(),
            'mlp_intermediate_states': OnlineStats()
        }

    def get_layer_hooks(self, layer_idx, layer):
        """
        为每个模块返回 Hook 函数，更新统计激活值。
        这里不再传递权重给 update，只聚焦激活值统计。
        """
        def q_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['attention_input_states'].update(
                input[0].detach().cpu()
            )

        def o_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['attention_post_aggregation'].update(
                input[0].detach().cpu()
            )

        def gate_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['mlp_input_states'].update(
                input[0].detach().cpu()
            )

        def down_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['mlp_intermediate_states'].update(
                input[0].detach().cpu()
            )

        return q_proj_hook, o_proj_hook, gate_proj_hook, down_proj_hook

    def register_activation_hooks(self, model):
        """为模型每一层注册 Hook，用于收集激活值。"""
        self.layer_activations.clear()
        for i, layer in enumerate(model.model.layers):
            self.layer_activations[i] = self._init_stats_dict()
            q_hook, o_hook, g_hook, d_hook = self.get_layer_hooks(i, layer)
            layer.self_attn.q_proj.register_forward_hook(q_hook)
            layer.self_attn.o_proj.register_forward_hook(o_hook)
            layer.mlp.gate_proj.register_forward_hook(g_hook)
            layer.mlp.down_proj.register_forward_hook(d_hook)

    def clear_activations(self):
        """清空当前收集的激活统计数据。"""
        for layer_idx in self.layer_activations:
            for key in self.layer_activations[layer_idx]:
                self.layer_activations[layer_idx][key].reset()


def save_activations(model, tokenizer, hook_manager, shot_inputs, task_types, output_root='../activations'):
    """
    针对不同任务类型的 prompts，调用模型进行推理并通过 Hook 收集激活值，最后保存到文件。
    """
    os.makedirs(output_root, exist_ok=True)

    # 按任务类型分组 prompts
    task_to_prompts = {}
    for prompt, ttype in zip(shot_inputs, task_types):
        task_to_prompts.setdefault(ttype, []).append(prompt)

    model.eval()
    with torch.no_grad():
        for ttype, prompts in task_to_prompts.items():
            # 每次处理前先清空统计数据
            hook_manager.clear_activations()

            for prompt in tqdm(prompts, desc=f"Processing {ttype}", unit="prompt"):
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                _ = model(**inputs)

                # 释放临时变量占用的显存
                del inputs
                gc.collect()
                torch.cuda.empty_cache()

            # 获取统计结果
            final_dict = {}
            for layer_idx, keys_dict in hook_manager.layer_activations.items():
                final_dict[layer_idx] = {}
                for key, stats_obj in keys_dict.items():
                    stats = stats_obj.get_stats()
                    final_dict[layer_idx][key] = {
                        "mean": stats["mean"].float(),
                        "var":  stats["var"].float(),
                        "l2":   stats["l2"].float()
                    }

            # 清空 hook_manager 统计数据
            hook_manager.clear_activations()
            gc.collect()
            torch.cuda.empty_cache()

            # 保存结果
            task_dir = os.path.join(output_root, ttype)
            os.makedirs(task_dir, exist_ok=True)
            save_path = os.path.join(task_dir, 'activations.pt')
            torch.save(final_dict, save_path)
            print(f"Saved activations for task type: {ttype} to {save_path}")

    gc.collect()
    torch.cuda.empty_cache()

def load_activations(root_path='../activations'):
    """
    从给定目录加载所有已保存的激活数据，并按任务类型组织返回。
    """
    task_to_activations = {}
    if not os.path.exists(root_path):
        return task_to_activations

    for ttype in os.listdir(root_path):
        task_dir = os.path.join(root_path, ttype)
        if os.path.isdir(task_dir):
            print('Loading:',ttype)
            activations_file = os.path.join(task_dir, 'activations.pt')
            if os.path.exists(activations_file):
                loaded_acts = torch.load(activations_file)
                task_to_activations[ttype] = loaded_acts
    return task_to_activations

def compute_and_save_weight_l2(model, save_path):
    """
    只需一次性地计算模型各层关心的权重的L2范数，并保存为文件。
    按 FLAP 方式，对 o_proj (input channels) 与 down_proj (input channels) 做列维度的 L2 汇总。
    """
    weight_l2_info = {}
    for i, layer in enumerate(model.model.layers):
        layer_dict = {}

        # o_proj.weight.shape = [hidden_size, hidden_size] (out_features, in_features)
        # sum(dim=0) -> shape=[in_features=hidden_size]
        o_l2 = (layer.self_attn.o_proj.weight ** 2).sum(dim=0).cpu()

        # down_proj.weight.shape = [hidden_size, intermediate_size]
        # sum(dim=0) -> shape=[in_features=intermediate_size]
        down_l2 = (layer.mlp.down_proj.weight ** 2).sum(dim=0).cpu()

        layer_dict['o_proj'] = o_l2
        layer_dict['down_proj'] = down_l2

        weight_l2_info[i] = layer_dict

    torch.save(weight_l2_info, save_path)
    print(f"Saved weight L2 info to {save_path}")


def load_weight_l2_info(weight_l2_file):
    """
    从指定文件加载模型各层的权重L2范数信息。
    返回一个嵌套字典：
      {layer_idx: {'o_proj': tensor_of_l2, 'down_proj': tensor_of_l2, ...}, ...}
    """
    if not os.path.exists(weight_l2_file):
        raise FileNotFoundError(f"Weight L2 info file not found: {weight_l2_file}")
    weight_l2_data = torch.load(weight_l2_file)
    print(f"Loaded weight L2 info from {weight_l2_file}")
    return weight_l2_data


def aggregate_task_activations(task_activations):
    """
    构造示例性聚合函数，将每个任务的激活数据拆成 Subset_1 和 Subset_2。
    你也可以按需对数据进行进一步处理或聚合。
    """
    aggregated_data = {}
    for ttype, layer_data in task_activations.items():
        aggregated_data[ttype] = {}
        aggregated_data[ttype]['Subset_1'] = {}
        aggregated_data[ttype]['Subset_2'] = {}
        for layer_idx, module_data in layer_data.items():
            aggregated_data[ttype]['Subset_1'][layer_idx] = {}
            aggregated_data[ttype]['Subset_2'][layer_idx] = {}
            for module_name, stats_dict in module_data.items():
                aggregated_data[ttype]['Subset_1'][layer_idx][module_name] = stats_dict
                aggregated_data[ttype]['Subset_2'][layer_idx][module_name] = stats_dict
    return aggregated_data


def plot_selected_neurons_activations(
    activations_data,
    layers_to_plot,
    neuron_indices,
    task_indices=None,
    subsets=['Subset_1', 'Subset_2'],
    module_name='attention_input_states',
    plot_field='mean',
    fontsize=48,
    tick_fontsize=36,
    cbar_fontsize=36,        # 色条相关字体大小
    random_seed=None,
    hspace=0.6,              # 子图之间的高度间距
    wspace=0.3,              # 子图之间的宽度间距
    normalize=True           # 新增：是否进行 z-score 标准化
):
    """
    绘制激活值热力图的示例函数，支持选择指定层、指定神经元、指定任务进行展示。
    通过 normalize 控制是否对数据进行 z-score 标准化。
    """

    def z_score_standardize(data):
        std_val = np.std(data)
        if std_val == 0:
            return data - np.mean(data)
        return (data - np.mean(data)) / std_val

    task_types = list(activations_data.keys())
    if task_indices is None:
        task_indices = range(len(task_types))
    selected_task_types = [task_types[i] for i in task_indices]

    # 如果 neuron_indices 是整数，随机挑选指定数量的神经元
    if isinstance(neuron_indices, int):
        num_neurons_to_select = neuron_indices
        found_neuron_count = None
        for ttype in selected_task_types:
            for subset_name in subsets:
                layer_data = activations_data[ttype].get(subset_name, {})
                for layer_idx in layers_to_plot:
                    data_dict = layer_data.get(layer_idx, {}).get(module_name, {})
                    if isinstance(data_dict, dict) and plot_field in data_dict:
                        found_neuron_count = len(data_dict[plot_field])
                        break
                if found_neuron_count is not None:
                    break
            if found_neuron_count is not None:
                break
        if found_neuron_count is None:
            raise ValueError("无法找到可用的神经元数据。")
        if num_neurons_to_select > found_neuron_count:
            raise ValueError(f"请求的神经元数量 {num_neurons_to_select} 超出可用神经元总数 {found_neuron_count}。")
        if random_seed is not None:
            random.seed(random_seed)
        neuron_indices = random.sample(range(found_neuron_count), num_neurons_to_select)

    num_task_types = len(selected_task_types)
    num_subsets = len(subsets)

    # 创建子图
    fig, axes = plt.subplots(num_subsets, num_task_types,
                             figsize=(len(neuron_indices) * num_task_types,
                                      len(layers_to_plot) * num_subsets * 2))
    # 调整子图间的间距
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # 处理 axes 的形状，保证可以通过 axes[row_idx, col_idx] 索引
    if num_subsets == 1 and num_task_types == 1:
        axes = np.array([[axes]])
    elif num_subsets == 1:
        axes = axes[np.newaxis, :]
    elif num_task_types == 1:
        axes = axes[:, np.newaxis]

    for col_idx, task_type in enumerate(selected_task_types):
        module_data_for_task = activations_data[task_type]
        for row_idx, subset_name in enumerate(subsets):
            layer_data = module_data_for_task.get(subset_name, {})
            data_matrix = []
            for layer_idx in layers_to_plot:
                data_dict = layer_data.get(layer_idx, {}).get(module_name, {})
                if not isinstance(data_dict, dict) or plot_field not in data_dict:
                    data_matrix.append(np.zeros(len(neuron_indices)))
                    continue
                neuron_activations = data_dict[plot_field]
                if len(neuron_activations) == 0:
                    data_matrix.append(np.zeros(len(neuron_indices)))
                    continue
                if max(neuron_indices) >= len(neuron_activations):
                    raise IndexError(f"Neuron indices {neuron_indices} exceed dimension {len(neuron_activations)}")
                
                # 拿到所选神经元的激活值
                selected_neurons = neuron_activations[neuron_indices].cpu().numpy()
                
                # 是否进行 z-score 标准化
                if normalize:
                    selected_neurons = z_score_standardize(selected_neurons)

                data_matrix.append(selected_neurons)

            data_matrix = np.array(data_matrix)
            layer_labels = [f'L {idx}' for idx in layers_to_plot]
            neuron_labels = [f'D {idx}' for idx in neuron_indices]
            ax = axes[row_idx, col_idx]

            # 绘制热力图
            heatmap = sns.heatmap(
                data_matrix,
                xticklabels=neuron_labels,
                yticklabels=layer_labels,
                cmap='coolwarm',
                center=0 if normalize else None,  # 若归一化，居中；否则根据数据本身范围
                ax=ax,
                cbar=True,
                cbar_kws={
                    'shrink': 0.8,
                    'label': f'Activation {plot_field}'
                }
            )
            # 调整色条的字体
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_fontsize)      # 刻度字体
            cbar.set_label(f'Activation {plot_field}', size=cbar_fontsize)  # 标签字体

            # 设置标题与坐标轴标签
            ax.set_title(f'{task_type} - {subset_name}', fontsize=fontsize, fontweight='bold', pad=20)
            ax.set_xlabel('Dimensions', fontsize=fontsize, labelpad=10)
            ax.set_ylabel('Layers', fontsize=fontsize, labelpad=10)
            ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

    plt.show()
