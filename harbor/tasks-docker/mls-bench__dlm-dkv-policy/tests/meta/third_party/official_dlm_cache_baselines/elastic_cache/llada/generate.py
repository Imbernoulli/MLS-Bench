# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
import math
from einops import rearrange
from copy import deepcopy

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise



@ torch.no_grad()
def generate_with_elastic_cache(
    model, prompt, gen_length=512, window_length=16, 
    mask_id=126336, eos_id=126081,
    threshold=0.9, tokens_per_iter=1,
    gamma=0.9, track_num=1, block_caching=True,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        gen_length: Generated answer length.
        window_length: Sliding window length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        mask_id: The toke id of [MASK] is 126336.
        eos_id: The toke id of [EOS] is 126081.
        threshold: confident-aware decoding.
        token_per_iter: top-k fixed decoding.
        gamma: cache update trigger threshold.
        track_num: number of most-attended tokens used for cache update trigger.
        block_caching: block caching far-away [MASK] tokens.
    '''

    # num_blocks = gen_length // block_length
    # assert steps % num_blocks == 0
    # steps = steps // num_blocks

    for l, block in enumerate(model.model.transformer.blocks):
        block.x_cache = None
        block.q_cache = None
        block.k_cache = None
        block.v_cache = None
        block.track_token = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()


    nfe = 0  
    query_position = torch.arange(prompt.shape[1] + gen_length, device=model.device)
    track_position = query_position[:0].clone()
    new_decoded_position = query_position[:prompt.shape[1]].clone()
    masked_position = query_position[prompt.shape[1]:].clone()

    i = 0
    decoded_eos = False
    num_computed = 0
    total_computed = 0
    L = len(model.model.transformer.blocks)

    while True:    
        if block_caching:
            query_masked_position = masked_position[:window_length]
        else:
            query_masked_position = masked_position

        if i == 0:
            x_query = x
            start_reset = -1
        else:
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query = x[:, query_position]
            start_reset = L

        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths = [x.shape[1], start_reset, gamma, track_num]

        output = model(x_query, use_cache=True, lengths=lengths, positions=positions)
        logits = output.logits

        if logits.shape[1] == x.shape[1]:
            logits = logits[:, query_masked_position, :]
        else:
            logits = logits[:, -query_masked_position.shape[0]:, :]

        if not block_caching:
            query_masked_position = query_masked_position[:window_length]
            logits = logits[:, :window_length]
        
        track_position = torch.cat([block.track_token for block in model.model.transformer.blocks], dim=0).unique(sorted=False)

        if threshold is not None:
            x, new_decoded_position, eos_pos = get_decoded_token_confident(logits, query_masked_position, x, threshold, eos_id)
        else:
            x, new_decoded_position, eos_pos = get_decoded_token_topk(logits, query_masked_position, x, tokens_per_iter, eos_id)
            
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]
        
        nfe += 1

        if not decoded_eos:
            if eos_pos.shape[0] > 0:
                eos_pos = eos_pos.min().item()
                decoded_eos = True
                masked_position = masked_position[masked_position <= eos_pos]

        num_computed += L - lengths[1]
        total_computed += L
        
        if masked_position.shape[0] == 0:
            break

        i += 1
    return x, nfe, num_computed / total_computed


def get_decoded_token_confident(logits, query_masked_position, x, threshold=None, eos_id=126081):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p, x0 = torch.max(p, dim=-1) # b, l
    keep_idx = (x0_p >= min(threshold, x0_p.max()))
    decoded_position = query_masked_position[keep_idx[0]]
    x0 = x0[:, keep_idx[0]]
    x[:, decoded_position] = x0
    return x, decoded_position, decoded_position[x0.eq(eos_id)[0]]  

def get_decoded_token_topk(logits, query_masked_position, x, num_transfer_tokens=None, eos_id=126081):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p, x0 = torch.max(p, dim=-1) # b, l
    _, keep_idx = torch.topk(x0_p[0], k=min(num_transfer_tokens, x0.shape[1]), largest=True)
    decoded_position = query_masked_position[keep_idx]
    x0 = x0[:, keep_idx]
    x[:, decoded_position] = x0
    return x, decoded_position, decoded_position[x0.eq(eos_id)[0]]  


def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_elastic_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
