import pickle
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
from features import FEATURESV30
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import gc

import pyarrow.parquet as pq
import pyarrow as pa

def clear_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def lerobot_to_pickle(repo_id, output_path, chunk_size=50, home_lerobot=None, batch_size=32, num_workers=0, save_interval=10):
    """
    将lerobot数据集转换为pickle文件（优化版本 - 分批保存并彻底清理内存）
    
    参数:
        repo_id: lerobot数据集的repo_id
        output_path: 输出pickle文件的路径
        chunk_size: 动作序列的长度（默认为50，当前未使用）
        home_lerobot: lerobot数据集的本地路径（如果为None，则使用默认路径）
        batch_size: 批处理大小（默认32）
        num_workers: 并行加载数据的进程数（默认0，单进程避免共享内存问题）
        save_interval: 每处理多少个batch后检查并保存已完成的episode（默认10）
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建数据集
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(os.path.join(home_lerobot, repo_id)),
        revision="main",
        download_videos=False,
        force_cache_sync=False,
        video_backend="pyav"
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 使用 DataLoader 进行批处理和并行加载
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False if num_workers == 0 else True
    )
    
    # 按 episode 组织数据
    episode_data = defaultdict(list)
    saved_episodes = set()  # 记录已保存的episode
    seen_episodes = set()    # 记录所有见过的episode（用于跨batch检查）
    processed_count = 0
    batch_count = 0
    
    def save_episode(episode_idx):
        """保存单个episode并彻底清理内存"""
        if episode_idx not in saved_episodes and episode_idx in episode_data:
            episode_file = os.path.join(output_path, f'episode_{episode_idx}.pkl')
            
            # 保存episode数据
            with open(episode_file, 'wb') as f:
                pickle.dump(episode_data[episode_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
            
            saved_episodes.add(episode_idx)
            
            # 彻底清理这个episode的内存
            # 1. 先清理episode数据中的tensor
            for sample in episode_data[episode_idx]:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        del value
                    elif isinstance(value, np.ndarray):
                        del value
            
            # 2. 删除整个episode数据
            del episode_data[episode_idx]
            
            # 3. 强制垃圾回收
            clear_memory()
            
            return True
        return False
    
    # 批量处理数据
    prev_episode_idx = None
    for batch in tqdm.tqdm(dataloader, desc="处理数据"):
        batch_size_actual = len(batch['episode_index'])
        batch_count += 1
        
        # 获取当前batch中出现的所有episode
        current_batch_episodes = set()
        for i in range(batch_size_actual):
            episode_idx = batch['episode_index'][i].item()
            current_batch_episodes.add(episode_idx)
            seen_episodes.add(episode_idx)
        
        # 处理批次中的每个样本
        for i in range(batch_size_actual):
            episode_idx = batch['episode_index'][i].item()
            
            # 如果遇到新的episode，立即保存上一个episode（在batch内部切换）
            if prev_episode_idx is not None and episode_idx != prev_episode_idx:
                if save_episode(prev_episode_idx):
                    print(f"已保存 episode_{prev_episode_idx}.pkl (共 {len(saved_episodes)} 个episode已保存，内存已清理)")
            
            prev_episode_idx = episode_idx
            
            # 处理样本，直接保存 tensor（不转换为 numpy）
            processed_sample = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    sample_tensor = value[i]
                    if sample_tensor.is_cuda:
                        processed_sample[key] = sample_tensor.cpu()
                    else:
                        # 创建tensor的副本，避免引用原始batch数据
                        processed_sample[key] = sample_tensor.clone()
                elif isinstance(value, (list, tuple)):
                    processed_sample[key] = value[i]
                elif isinstance(value, np.ndarray):
                    if value.ndim > 0:
                        # 创建numpy数组的副本，避免引用原始batch数据
                        processed_sample[key] = value[i].copy()
                    else:
                        processed_sample[key] = value
                else:
                    processed_sample[key] = value
            
            episode_data[episode_idx].append(processed_sample)
            processed_count += 1
        
        # 清理当前batch的引用，帮助释放内存
        del batch
        
        # 定期检查：保存那些在之前batch中出现过，但在当前batch中没有出现的episode
        if batch_count % save_interval == 0:
            episodes_to_save = []
            for ep_idx in list(episode_data.keys()):
                if ep_idx not in current_batch_episodes and ep_idx in seen_episodes:
                    if len(current_batch_episodes) > 0:
                        min_current_ep = min(current_batch_episodes)
                        if ep_idx < min_current_ep:
                            episodes_to_save.append(ep_idx)
            
            # 保存这些已完成的episode
            for ep_idx in episodes_to_save:
                if save_episode(ep_idx):
                    print(f"定期保存 episode_{ep_idx}.pkl (共 {len(saved_episodes)} 个episode已保存，内存已清理)")
            
            # 显示当前内存中的episode数量
            if len(episode_data) > 0:
                print(f"当前内存中有 {len(episode_data)} 个episode的数据，已保存 {len(saved_episodes)} 个episode")
            
            # 定期清理内存
            clear_memory()
    
    # 保存最后一个episode（如果还没有保存）
    if prev_episode_idx is not None and prev_episode_idx not in saved_episodes:
        if save_episode(prev_episode_idx):
            print(f"已保存最后一个 episode_{prev_episode_idx}.pkl")
    
    # 保存所有剩余的episode（确保没有遗漏）
    if len(episode_data) > 0:
        print(f"\n保存剩余的 {len(episode_data)} 个episode...")
        for episode_idx in sorted(episode_data.keys()):
            if save_episode(episode_idx):
                print(f"已保存 episode_{episode_idx}.pkl")
    
    # 最终清理
    del episode_data
    clear_memory()
    
    print(f"\n数据已保存到: {output_path}")
    print(f"总共保存了 {len(saved_episodes)} 个 episode")
    print(f"总共处理了 {processed_count} 个样本")
    
    # 验证保存的文件
    saved_files = [f for f in os.listdir(output_path) if f.endswith('.pkl')]
    print(f"实际保存的文件数: {len(saved_files)}")
    
    # 验证数据完整性
    if len(saved_episodes) != len(saved_files):
        print(f"⚠️  警告: 保存的episode数 ({len(saved_episodes)}) 与文件数 ({len(saved_files)}) 不匹配！")

# 使用示例
if __name__ == "__main__":
    lerobot_to_pickle(
        "folding_clothes", 
        "tmp_data", 
        home_lerobot="/home/congcong/.cache/huggingface/lerobot",
        batch_size=64,
        num_workers=16,      # 单进程避免共享内存问题
        save_interval=10    # 每10个batch检查一次（可以根据需要调整）
    )