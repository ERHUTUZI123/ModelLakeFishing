import torch

from ModelLakeFishing.stage2TrainGraphSAGE.sampling import build_csr


def _all_neighbours(csr, nodes):
    ptr, nbr = csr
    if nodes.numel() == 0:
        return nodes.new_zeros(0)
    counts = ptr[nodes + 1] - ptr[nodes]
    total = int(counts.sum())
    if total == 0:
        return nodes.new_zeros(0)
    offs = (torch.arange(total)
            - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            + torch.repeat_interleave(ptr[nodes], counts))
    return nbr[offs]


def _closure(data, seeds, num_hops, csr_cache):
    visited = {t: torch.zeros(data[t].num_nodes, dtype=torch.bool)
               for t in data.node_types}
    frontier = {t: torch.zeros(0, dtype=torch.long) for t in data.node_types}
    for t, v in seeds.items():
        visited[t][v] = True
        frontier[t] = v
    for _ in range(num_hops):
        found = {t: [] for t in visited}
        for et in data.edge_types:
            st, _rel, dt = et
            fwd, bwd = csr_cache[et]
            found[dt].append(_all_neighbours(fwd, frontier[st]))
            found[st].append(_all_neighbours(bwd, frontier[dt]))
        grew = False
        for t in visited:
            cat = torch.cat(found[t]) if found[t] else torch.zeros(0, dtype=torch.long)
            new = torch.unique(cat)
            new = new[~visited[t][new]] if new.numel() else new
            visited[t][new] = True
            frontier[t] = new
            grew |= bool(new.numel())
        if not grew:
            break
    return {t: v.nonzero().flatten() for t, v in visited.items()}


@torch.no_grad()
def chunked_forward(model, data, *, chunk_size=50_000, device="cpu",
                    num_hops=None, out_device="cpu", progress=False):
    model.eval()
    if num_hops is None:
        num_hops = getattr(model, "num_layers", 2)
    n_nodes = {t: data[t].num_nodes for t in data.node_types}
    csr_cache = {et: (build_csr(data[et].edge_index, n_nodes[et[0]]),
                      build_csr(data[et].edge_index.flip(0), n_nodes[et[2]]))
                 for et in data.edge_types}

    out = {}
    for nt in data.node_types:
        N = n_nodes[nt]
        for start in range(0, N, chunk_size):
            seed = torch.arange(start, min(start + chunk_size, N), dtype=torch.long)
            ids = _closure(data, {nt: seed}, num_hops, csr_cache)
            for t in list(ids):
                if ids[t].numel() == 0:
                    ids[t] = torch.zeros(1, dtype=torch.long)
            sub = data.subgraph(dict(ids))
            z = model(sub.clone().to(device))
            g2l = torch.full((N,), -1, dtype=torch.long)
            g2l[ids[nt]] = torch.arange(ids[nt].numel())
            local = g2l[seed]
            assert int(local.min()) >= 0, "seed node missing from its own closure"
            block = z[nt][local.to(z[nt].device)].to(out_device)
            if nt not in out:
                out[nt] = torch.empty((N, block.size(1)), dtype=block.dtype,
                                      device=out_device)
            out[nt][seed] = block
            if progress:
                print(f"    [chunked_forward] {nt} {min(start + chunk_size, N)}/{N} "
                      f"(closure {sum(v.numel() for v in ids.values())} nodes)",
                      flush=True)
    return out
