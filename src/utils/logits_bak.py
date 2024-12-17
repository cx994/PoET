import torch
import argparse
import itertools
import string
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from .config import config
from poet.alphabets import Uniprot21
from poet.fasta import parse_stream
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET
from poet.msa.sampling import MSASampler, NeighborsSampler
import pytorch_lightning as pl


ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
PBAR_POSITION = 1


T = TypeVar("T", np.ndarray, torch.Tensor)


def append_startstop(x: T, alphabet: Uniprot21) -> T:
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
    )


def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.cuda()
        with torch.no_grad():
            memory = model.logits_allocate_memory(
                memory = memory,
                batch_size = batch_size, 
                length = max_variant_length - 1
            )
            
            try:
                # 使用预分配的内存进行计算
                logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)
            finally:
                # 确保内存被释放
                if memory is not None:
                    for mem in memory:
                        mem.x = None  # 释放 tensor 内存
                    memory = None
                torch.cuda.empty_cache()
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.detach().cpu().numpy())
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )

