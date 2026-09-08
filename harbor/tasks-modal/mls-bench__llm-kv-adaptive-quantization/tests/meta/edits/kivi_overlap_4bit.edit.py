"""Source-backed KIVI overlap baseline aligned to the official repo defaults."""

_FILE = "transformers-kv-lab/custom_quant_eval.py"

_POLICY = """\
class AdaptiveKVQuantizer:
    \"\"\"KIVI K4/V4 axes with streaming residual behavior.\"\"\"

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = 4

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str) -> int:
        residual_length = max(0, min(seq_len, int(residual_length)))
        if residual_policy == "block_modulo":
            return seq_len % residual_length if residual_length else 0
        if residual_policy == "tail":
            return residual_length
        return 0

    def _minmax_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2**bits) - 1)
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

    def _quantize(self, tensor: torch.Tensor, axis: str, residual_policy: str) -> tuple[torch.Tensor, float]:
        work = tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual = self._residual_keep_length(seq_len, self.key_residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":
            usable = quant_slice.shape[-2] - (quant_slice.shape[-2] % self.group_size)
            if usable > 0:
                main = quant_slice[:, :, :usable, :].transpose(2, 3)
                main = main.reshape(batch, heads, head_dim, usable // self.group_size, self.group_size)
                main = self._minmax_last_dim(main, self.bits, self.group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            fp_tokens = residual + (quant_slice.shape[-2] - usable)
            avg_bits = (usable * self.bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_last_dim(flat, self.bits, self.group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(key_states, "channel", "block_modulo")

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(value_states, "token", "tail")

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        policy = "block_modulo" if kv_kind == "key" else "tail"
        residual = self._residual_keep_length(seq_len, self.key_residual_length, policy)
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
