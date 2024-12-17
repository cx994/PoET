import torch
import torch.nn.functional as F

def align_logits_dimensions(global_logits, local_logits, sequence):
    """
    将全局和局部的logits维度对齐
    
    Args:
        global_logits (torch.Tensor): ESM模型产生的全局logits
        local_logits (torch.Tensor): SDP模型产生的局部logits
        sequence (str): 原始蛋白质序列
    
    Returns:
        tuple: (aligned_global_logits, aligned_local_logits)
    """
    # 获取序列长度
    seq_len = len(sequence)
    
    # ESM tokenizer会在序列开始添加<cls>标记，在结尾添加<eos>标记
    # 去除global_logits的<cls>和<eos>标记对应的logits
    if global_logits.dim() == 2:  # [seq_len + 2, vocab_size]
        global_logits = global_logits[1:-1]  # 移除首尾token的logits
    
    # SDP的tokenizer直接对序列进行处理，不添加特殊标记
    # 确保local_logits的长度与序列长度匹配
    if local_logits.size(0) != seq_len:
        # 如果需要，可以使用插值来调整维度
        local_logits = F.interpolate(
            local_logits.unsqueeze(0).unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    
    # 确保两个logits具有相同的设备和数据类型
    if global_logits.device != local_logits.device:
        local_logits = local_logits.to(global_logits.device)
    
    if global_logits.dtype != local_logits.dtype:
        local_logits = local_logits.to(dtype=global_logits.dtype)
    
    return global_logits, local_logits

def combine_logits(global_logits, local_logits, alpha=0.5):
    """
    将对齐后的全局和局部logits进行加权组合
    
    Args:
        global_logits (torch.Tensor): 对齐后的全局logits
        local_logits (torch.Tensor): 对齐后的局部logits
        alpha (float): 全局logits的权重，局部logits的权重为(1-alpha)
    
    Returns:
        torch.Tensor: 组合后的logits
    """
    return alpha * global_logits + (1 - alpha) * local_logits 