"""MLS-Bench 4-bit SQuat-inspired subspace-orthogonal KV-cache quantizer."""

_FILE = "transformers-kv-lab/custom_quant_eval.py"

_POLICY = """\
class AdaptiveKVQuantizer:
    \"\"\"SQuat-inspired subspace-orthogonal K/V 4-bit quantization.\"\"\"

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.residual_length = 32
        self.subspace_dim = 60
        self.squat_lambda = 0.001
        self.quant_group_size = 64
        self.shared_svd = True
        self.query_subspaces = {}

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.query_subspaces = {}

    def needs_prefill_qkv_observer(self) -> bool:
        return True

    def query_observation_position(self) -> str:
        return "post_rope"

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        if query_states is None:
            return None
        batch, query_heads, _, head_dim = query_states.shape
        kv_heads = int(attention_meta.get("kv_heads", query_heads))
        if query_heads % kv_heads != 0:
            kv_heads = query_heads
        matrix = query_states.reshape(batch, kv_heads, -1, head_dim).float()
        rank = min(int(self.subspace_dim), matrix.shape[-2], matrix.shape[-1])
        if rank <= 0:
            return None
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        scaled_vh = torch.diag_embed(singular_values[:, :, :rank]).matmul(vh[:, :, :rank, :])
        self.query_subspaces[layer_id] = (scaled_vh[0:1] if self.shared_svd else scaled_vh).detach()
        return None

    def _residual_keep_length(self, seq_len: int) -> int:
        residual_length = max(0, min(seq_len, int(self.residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _minmax_last_dim(self, data: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, 2**int(bits) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _generate_At_inv(self, query_subspace: torch.Tensor, tol: float = 1e-7):
        batch, heads, _, head_dim = query_subspace.shape
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        matrices = [None] * groups
        eye = torch.eye(head_dim, device=query_subspace.device, dtype=torch.float32)
        A_t = eye.expand(batch, heads, head_dim, head_dim) + float(self.squat_lambda) * query_subspace.float().transpose(
            -1, -2
        ).matmul(query_subspace.float())
        matrices[groups - 1] = A_t
        for group_idx in range(groups - 1, 0, -1):
            current_dim = group_idx * q_group
            width = min(q_group, A_t.shape[-1] - current_dim)
            M_t1 = A_t[:, :, :current_dim, :current_dim]
            N_t1 = A_t[:, :, current_dim : current_dim + width, :current_dim]
            O_t1 = A_t[:, :, current_dim : current_dim + width, current_dim : current_dim + width]
            local_eye = torch.eye(width, device=query_subspace.device, dtype=torch.float32)
            O_inv = torch.inverse(O_t1 + tol * local_eye.expand(batch, heads, width, width))
            A_t = M_t1 - N_t1.transpose(-1, -2).matmul(O_inv.matmul(N_t1))
            matrices[group_idx - 1] = A_t[:, :, :, -q_group:]
        return matrices

    def _squat_quantize_keys(self, key_states: torch.Tensor, query_subspace: torch.Tensor) -> torch.Tensor:
        batch, heads, _, head_dim = key_states.shape
        query_subspace = query_subspace.to(device=key_states.device)
        if query_subspace.shape[0] == 1 and batch > 1:
            query_subspace = query_subspace.expand(batch, -1, -1, -1)
        if query_subspace.shape[1] != heads or query_subspace.shape[-1] != head_dim:
            raise ValueError("SQuat query subspace shape does not match the key tensor")
        matrices = self._generate_At_inv(query_subspace)
        P_inv = torch.inverse(matrices[-1])
        work = key_states.float().clone()
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        for group_idx in range(groups):
            start = group_idx * q_group
            end = min(head_dim, start + q_group)
            chunk = work[:, :, :, start:end]
            dequant = self._minmax_last_dim(chunk.transpose(2, 3).contiguous(), self.group_size, self.bits).transpose(2, 3)
            if group_idx < groups - 1:
                d_vec = (dequant - chunk).float()
                next_start = end
                H_t = matrices[group_idx]
                B_t = P_inv[:, :, next_start:, :next_start]
                update = d_vec.matmul(H_t.transpose(-2, -1)).matmul(B_t.transpose(-2, -1))
                work[:, :, :, next_start:] = work[:, :, :, next_start:] + update
            work[:, :, :, start:end] = dequant
        return work

    def _quantize_with_residual(self, tensor: torch.Tensor, quant_fn) -> tuple[torch.Tensor, float]:
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
        avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        query_subspace = self.query_subspaces.get(layer_id)
        if query_subspace is None:
            raise RuntimeError("SQuat key quantization requires the prefill query observer")
        return self._quantize_with_residual(key_states, lambda data: self._squat_quantize_keys(data, query_subspace))

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_with_residual(value_states, lambda data: self._minmax_last_dim(data, self.group_size, self.bits))

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        residual = self._residual_keep_length(seq_len)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 41,
        "end_line": 172,
        "content": _POLICY,
    },
]
