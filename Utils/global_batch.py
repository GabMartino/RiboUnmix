"""GPU-count-independent logical batches, partitioned at transcript boundaries."""


def partition_execution_groups(groups, lengths, *, world_size, reference_count,
                               max_groups=None, max_pairs=None, max_tokens=None):
    """Return rank-local chunks and aligned idle slots; never duplicate data.

    The caller constructs the logical batch before this function. Greedy cost
    balancing accounts for observed pairs and the full fixed reference. An
    oversized transcript remains atomic, as in the single-process sampler.
    """
    ranks = [[] for _ in range(world_size)]
    costs = [0] * world_size
    for group in groups:
        rank = min(range(world_size), key=lambda r: (costs[r], r))
        ranks[rank].append(group)
        costs[rank] += max(int(lengths[i]) for i in group) * (len(group) + reference_count)
    plans = []
    for assigned in ranks:
        chunks, chunk, pairs, length = [], [], 0, 0
        for group in assigned:
            new_pairs = pairs + len(group)
            new_length = max(length, max(int(lengths[i]) for i in group))
            full = ((max_groups is not None and len(chunk) >= max_groups)
                    or (max_pairs is not None and new_pairs > max_pairs)
                    or (max_tokens is not None and new_pairs * new_length > max_tokens))
            if chunk and full:
                chunks.append(chunk)
                chunk, pairs, length = [], 0, 0
            chunk.append(group)
            pairs += len(group)
            length = max(length, max(int(lengths[i]) for i in group))
        if chunk:
            chunks.append(chunk)
        plans.append(chunks)
    slots = max(map(len, plans))
    return [chunks + [[] for _ in range(slots - len(chunks))] for chunks in plans]


def empty_execution_metadata(batch):
    if isinstance(batch, dict) and set(batch) == {'empty_execution'}:
        return batch['empty_execution']
    return None
