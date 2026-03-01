import math

import torch
import torch.nn.functional as F

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard


@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, S = q.shape
    o = torch.empty(B, T, H, S, dtype=q.dtype, device=q.device)
    final_state = torch.empty(B, H, S, S, dtype=q.dtype, device=q.device) if output_final_state else None
    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        o, final_state = chunk_quasar_fwd(
            q=q, k=k, v=v, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        return (torch.zeros_like(do), torch.zeros_like(do),
                torch.zeros_like(do), None,
                None, None, None)


@torch.compiler.disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return ChunkQuasarFunction.apply(
        q, k, v, beta, initial_state, output_final_state, cu_seqlens)
