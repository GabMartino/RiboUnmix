"""Execution-only starting points; these are heuristics, not measured optima."""
from __future__ import annotations


def execution_profile(n: int, memory_gib: float, profile: str = 'auto') -> dict:
    if n < 2 or memory_gib <= 0 or profile not in {'auto', 'aggressive'}:
        raise ValueError('Expected N >= 2, positive GPU memory, and auto/aggressive profile.')
    # A larger N already increases pair rows per transcript. Increase the
    # execution budget only on GPUs with sufficient memory; retain whole groups.
    rows, tokens, chunk = 512, 256000, 16
    if memory_gib < 30:
        rows, tokens, chunk = 256, 128000, 8
    elif n >= 40 and memory_gib >= 60:
        rows, tokens, chunk = 1024, 512000, 32
        if profile == 'aggressive' and memory_gib >= 75:
            rows, tokens, chunk = 2048, 1024000, 64
    return dict(
        max_pair_rows_per_forward=rows,
        max_padded_codon_tokens_per_forward=tokens,
        reference_chunk_size=min(n, chunk),
        log_every_n_steps=100,
        # Spawn workers copy the in-memory dataset. Avoid multiplying host
        # memory use at large N. This can be overridden after measuring stalls.
        data_num_workers=0 if n >= 40 else 2,
    )
