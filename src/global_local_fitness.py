from src.utils.dimension_alignment import align_logits_dimensions, combine_logits
from src.ESM_embedding import get_esm_logits
from src.SDP_score import get_sdp_logits

def calculate_combined_fitness(sequence, esm_model, sdp_model, alpha=0.5):
    """
    计算序列的组合fitness分数
    
    Args:
        sequence (str): 输入的蛋白质序列
        esm_model: ESM模型实例
        sdp_model: SDP模型实例
        alpha (float): 全局分数的权重
    
    Returns:
        torch.Tensor: 组合后的fitness分数
    """
    # 获取全局和局部的logits
    global_logits = get_esm_logits(sequence, esm_model)
    local_logits = get_sdp_logits(sequence, sdp_model)
    
    # 对齐维度
    aligned_global, aligned_local = align_logits_dimensions(
        global_logits, 
        local_logits, 
        sequence
    )
    
    # 组合logits
    combined_logits = combine_logits(aligned_global, aligned_local, alpha)
    
    return combined_logits
