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
from config import config
from poet.alphabets import Uniprot21
from poet.fasta import parse_stream
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET
from poet.msa.sampling import MSASampler, NeighborsSampler
from utils.logits import *
import pytorch_lightning as pl


ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
PBAR_POSITION = 1


T = TypeVar("T", np.ndarray, torch.Tensor)


msa_sub1_path = Path(config.msa_sub1_path)
msa_sub2_path = Path(config.msa_sub2_path)
variants_fasta_path = Path(config.variants_fasta_path)
output_npy_path = Path(config.output_npy_path)
batch_size = config.batch_size
max_similarity = config.max_similarity
max_tokens = config.max_tokens
ckpt = torch.load(config.ckpt_path)
seed = config.seed
pl.seed_everything(config.seed)


@torch.inference_mode()
def main():

    model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict({k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()})
    model = model.cuda().half().eval()
    alphabet = Uniprot21(
        include_gap=True, include_startstop=True, distinct_startstop=True
    )

    # get variants to score
    variants = [v for v in get_seqs_from_fastalike(variants_fasta_path)]
    variants = [seq.replace(b"*",b"A") for seq in variants]
    variants = [
        append_startstop(alphabet.encode(v), alphabet=alphabet)
        for v in variants
    ]
    logps = []
    max_tokens, max_similarity = (config.max_tokens,config.max_similarity)
    sampler = MSASampler(
        method=NeighborsSampler(
            can_use_torch=False,
        ),
        max_similarity=max_similarity,
    )

    msa_sequence_path = msa_sub1_path
    msa_sequences = get_seqs_from_fastalike(msa_sequence_path)
    msa_sequences = [seq.replace(b'*', b'-') for seq in msa_sequences]
    msa_sequences = msa_sequences
    msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa_sequences, alphabet=alphabet)
    n_eff, p = sampler.method.get_weights(msa=msa, gap_token=alphabet.gap_token)
    sample_idxs = sampler.get_sample_idxs(
        msa=msa,
        gap_token=alphabet.gap_token,
        seed=seed,
    )

    this_msa_sequences = sample_msa_sequences(
        get_sequence_fn=lambda ii: msa_sequences[ii]
        .upper()
        .translate(None, delete=b"-"),
        sample_idxs=sample_idxs,
        max_tokens=max_tokens,
        alphabet=alphabet,
        shuffle_seed=seed,
        truncate=False,
    )

    pbar_position = PBAR_POSITION
    segment_sizes = torch.tensor([len(s) for s in this_msa_sequences]).cuda()
    this_msa_sequences = torch.cat(
        [torch.from_numpy(s).long() for s in this_msa_sequences]
    ).cuda()
    memory = model.embed(
        this_msa_sequences.unsqueeze(0),
        segment_sizes.unsqueeze(0),
        allow_cpu_offload = True,
        pbar_position=pbar_position,
    )
    # memory
    this_variants = variants[0:batch_size]
    max_variant_length = len(this_variants)
    memory = model.logits_allocate_memory(
        memory = memory,
        batch_size = batch_size,
        length = max_variant_length -1
    )
    
    this_variants = pad_sequence(
                [torch.from_numpy(v).long() for v in this_variants],
                batch_first=True,
                padding_value=alphabet.mask_token,
            )

    assert (this_variants == alphabet.gap_token).sum() == 0
    this_variants = this_variants.cuda()

    logits2,embedding2 = model.logits(this_variants[:, :-1], memory, preallocated_memory=False,return_embeddings=True)
    