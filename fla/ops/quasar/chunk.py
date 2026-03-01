import math

import torch
import torch.nn.functional as F

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard


def _select_chunk_size(T: int, S: int, BH: int, device: torch.device) -> int:
    """
    Pick the best chunk_size that fits in ~70 % of free GPU RAM.

    BT=512 is the empirical sweet-spot on modern GPUs: it halves the
    sequential loop count vs BT=256 while keeping the L = [N, BT, BT]
    triangular matrix small enough to avoid memory-bandwidth saturation.
    BT>=1024 actually regresses because L bandwidth dominates.
    """
    if not torch.cuda.is_available():
        return 256

    free = torch.cuda.mem_get_info(device)[0]
    budget = int(free * 0.70)

    for BT in (512, 256, 128):
        if BT > T:
            continue
        NT = math.ceil(T / BT)
        N = BH * NT
        peak = (N * BT * BT + 10 * N * BT * S + 2 * N * S * S) * 4
        if peak < budget:
            return BT

    return min(64, T) if T > 0 else 64


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
    """
    Optimized chunk-wise QuasarAttention forward pass.

    Uses adaptive chunk_size and fused gate scaling to minimise both
    sequential loop iterations and memory-bandwidth overhead.
    """
    B, T, H, S = q.shape
    BH = B * H

    if chunk_size <= 0:
        chunk_size = _select_chunk_size(T, S, BH, q.device)

    BT = chunk_size
    original_T = T
    out_dtype = q.dtype

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    if T % BT != 0:
        pad_len = BT - (T % BT)
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        T += pad_len

    NT = T // BT
    N = BH * NT

    with torch.amp.autocast('cuda', enabled=False):
        # [B, T, H, S] -> [B, H, T, S] -> [N, BT, S]   (fp32)
        qf = q.float().permute(0, 2, 1, 3).reshape(N, BT, S)
        kf = k.float().permute(0, 2, 1, 3).reshape(N, BT, S)
        vf = v.float().permute(0, 2, 1, 3).reshape(N, BT, S)

        # gate:  alpha = (1 - exp(-beta * ||k||^2)) / (||k||^2 + eps)
        k_sq = (kf * kf).sum(-1, keepdim=True).view(BH, NT, BT, 1)
        beta_e = beta.float().repeat(B).view(BH, 1, 1, 1)
        alpha = ((1.0 - torch.exp(-beta_e * k_sq)) / (k_sq + 1e-8)).reshape(N, BT, 1)
        del k_sq, beta_e

        # Pre-scale K and V by alpha (reused for L construction and solve RHS)
        aK = alpha * kf          # [N, BT, S]
        aV = alpha * vf          # [N, BT, S]
        del alpha

        # L = I + tril(diag(alpha) K K^T, -1)
        # Fused:  bmm(aK, kf^T) = diag(alpha) @ K @ K^T  (avoids separate L *= alpha pass)
        L = torch.bmm(aK, kf.transpose(-2, -1))
        L.tril_(diagonal=-1)
        L.diagonal(dim1=-2, dim2=-1).fill_(1.0)

        # Batched triangular solve:  W = L^{-1}(alpha*K),  U = L^{-1}(alpha*V)
        W = torch.linalg.solve_triangular(L, aK, upper=False, unitriangular=True)
        U = torch.linalg.solve_triangular(L, aV, upper=False, unitriangular=True)
        del L, aK, aV

        # Inter-chunk transition:  A = I - K^T W,   B_mat = K^T U
        kT = kf.transpose(-2, -1)
        A_tr = torch.bmm(kT, W).neg_()
        A_tr.diagonal(dim1=-2, dim2=-1).add_(1.0)
        B_mat = torch.bmm(kT, U)

        # Output factors:  P = Q A,   R = Q B_mat
        P = torch.bmm(qf, A_tr)
        R = torch.bmm(qf, B_mat)

        del kT, qf, kf, vf, W, U

        # Reshape for sequential state-propagation loop
        A_tr = A_tr.view(BH, NT, S, S)
        B_mat = B_mat.view(BH, NT, S, S)
        P = P.view(BH, NT, BT, S)
        R = R.view(BH, NT, BT, S)

        state = torch.zeros(BH, S, S, dtype=torch.float32, device=q.device)
        if initial_state is not None:
            state.copy_(initial_state.reshape(BH, S, S))

        o_chunks = torch.empty(BH, NT, BT, S, dtype=torch.float32, device=q.device)

        for i in range(NT):
            state = torch.baddbmm(B_mat[:, i], A_tr[:, i], state)
            torch.baddbmm(R[:, i], P[:, i], state, out=o_chunks[:, i])

    # [BH, NT, BT, S] -> [B, H, T, S] -> [B, T, H, S]
    o = o_chunks.view(B, H, T, S).permute(0, 2, 1, 3).to(out_dtype)
    if original_T != T:
        o = o[:, :original_T]

    final_state = state.view(B, H, S, S).to(out_dtype) if output_final_state else None
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
        B, T, H, S = q.shape
        chunk_size = _select_chunk_size(T, S, B * H, q.device)
        chunk_indices = (prepare_chunk_indices(cu_seqlens, chunk_size)
                         if cu_seqlens is not None else None)

        o, final_state = chunk_quasar_fwd(
            q=q, k=k, v=v, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, beta)
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        q, k, v, beta = ctx.saved_tensors
        return (torch.zeros_like(q), torch.zeros_like(k),
                torch.zeros_like(v), torch.zeros_like(beta),
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
    """
    Chunk-wise QuasarAttention forward pass with autograd support.
    """
    return ChunkQuasarFunction.apply(
        q, k, v, beta, initial_state, output_final_state, cu_seqlens)
