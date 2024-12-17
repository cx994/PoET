"""
ESM tokenizer:  tokenizer._id_to_token

{0: '<cls>',
 1: '<pad>',
 2: '<eos>',
 3: '<unk>',
 4: 'L',
 5: 'A',
 6: 'G',
 7: 'V',
 8: 'S',
 9: 'E',
 10: 'R',
 11: 'T',
 12: 'I',
 13: 'D',
 14: 'P',
 15: 'K',
 16: 'Q',
 17: 'N',
 18: 'F',
 19: 'Y',
 20: 'M',
 21: 'H',
 22: 'W',
 23: 'C',
 24: 'X',
 25: 'B',
 26: 'U',
 27: 'Z',
 28: 'O',
 29: '.',
 30: '-',
 31: '<null_1>',
 32: '<mask>'}
"""


from transformers import EsmTokenizer, EsmForMaskedLM
import torch
from pathlib import Path
from typing import Union, List

def load_esm_model(model_path: str, device: str = None) -> tuple:
    """
    加载ESM模型和tokenizer
    
    Args:
        model_path (str): ESM模型路径
        device (str, optional): 计算设备. 默认为None，会自动选择
    
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def get_esm_logits(
    sequence: Union[str, List[str]], 
    model: EsmForMaskedLM, 
    tokenizer: EsmTokenizer = None,
    batch_size: int = 1
) -> torch.Tensor:
    """
    使用ESM模型计算序列的logits
    
    Args:
        sequence (Union[str, List[str]]): 输入序列或序列列表
        model (EsmForMaskedLM): ESM模型实例
        tokenizer (EsmTokenizer, optional): ESM tokenizer实例
        batch_size (int, optional): 批处理大小，默认为1
    
    Returns:
        torch.Tensor: 计算得到的logits，shape为[seq_len, vocab_size]
    """
    # 确保model处于评估模式
    model.eval()
    
    # 如果输入是单个序列，转换为列表
    if isinstance(sequence, str):
        sequences = [sequence]
    else:
        sequences = sequence
    
    if tokenizer is None:
        tokenizer = EsmTokenizer.from_pretrained(model.config._name_or_path)
    
    device = next(model.parameters()).device
    all_logits = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 如果是批处理大小为1，直接返回logits
            if len(batch_sequences) == 1:
                all_logits.append(logits.squeeze(0))
            else:
                # 否则，将每个序列的logits分别添加到列表中
                all_logits.extend(logits)
    
    # 如果输入是单个序列，返回单个结果
    if isinstance(sequence, str):
        return all_logits[0]
    
    return all_logits



    
    