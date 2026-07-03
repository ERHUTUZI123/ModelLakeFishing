"""
cold_config_manifest.py -- the SINGLE source of truth for the 23 Stage-2 versions
rerun in the cold-dataset evaluation. Flags reconstructed from the recorded run
commands (RESULTS_REPRODUCE.md, phase345/ranknet/phase67 scripts, top1 gphase),
NOT from memory. Every version is num_layers=1 (the cold-insertion invariant holds
for all). epochs 25 except P7_es / P7_lr3e3 (40, which defines those versions).

ACCEPTED_ranknet is excluded: it is only the single-seed checkpoint form of
R_mg02, not a distinct algorithmic version.
"""


def _base():
    return dict(num_layers=1, top_frac=0.10, lr=1e-2,
                lambda_rank=1.0, lambda_contrast=1.0, lambda_mse=0.0, lambda_uniform=0.0,
                scorer="dot", similar_to_mode="dense", similar_to_k=10,
                edge_aware=False, weighted_relations=None,
                rank_loss="hinge", rank_temperature=0.1, rank_min_gap=0.0, rank_gap_weighted=False,
                separate_heads=False, grouped=False,
                lambda_dm_contrast=0.0, dm_temperature=0.1, dm_hard_neg_weight=1.0, dm_warmup=0,
                early_stop=False, patience=0,
                lambda_global=0.0, global_known_low=False, global_hard_frac=0.0,
                epochs=25)


def _u(**kw):
    c = _base(); c.update(kw); return c


# topk edge-aware base shared by many rows
def _tk(**kw):
    c = _u(similar_to_mode="topk_unweighted", similar_to_k=10); c.update(kw); return c


def manifest():
    m = {}
    m["B0"] = _u()                                                     # dense, hinge
    m["B1"] = _u(similar_to_mode="none")
    m["B2"] = _u(similar_to_mode="topk_unweighted")
    m["B2e_ctrl"] = _tk(edge_aware=True, weighted_relations=[])
    m["B3_sim"] = _tk(edge_aware=True, weighted_relations=["similar_to"])
    m["B3_simtr"] = _tk(edge_aware=True, weighted_relations=["similar_to", "trained_on", "rev_trained_on"])
    m["B3_all"] = _tk(edge_aware=True, weighted_relations=None)
    m["B4_grouped"] = _tk(edge_aware=True, weighted_relations=[], grouped=True)
    m["B6_heads"] = _tk(edge_aware=True, weighted_relations=[], separate_heads=True)
    m["BEST"] = _tk(edge_aware=True, weighted_relations=None, rank_loss="ranknet",
                    rank_min_gap=0.01, separate_heads=True, grouped=True)
    m["B5_ranknet"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.01)
    m["R_tohet"] = _tk(edge_aware=False, rank_loss="ranknet", rank_min_gap=0.01)
    m["R_weights"] = _tk(edge_aware=True, weighted_relations=None, rank_loss="ranknet", rank_min_gap=0.01)
    m["R_heads"] = _tk(edge_aware=True, weighted_relations=[], separate_heads=True,
                       rank_loss="ranknet", rank_min_gap=0.01)
    m["R_mg00"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.0)
    m["R_mg02"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.02)
    m["P6_dm05"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet",
                       rank_min_gap=0.01, lambda_dm_contrast=0.5)
    m["P6_dm10"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet",
                       rank_min_gap=0.01, lambda_dm_contrast=1.0)
    m["P7_es"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet",
                     rank_min_gap=0.01, early_stop=True, patience=10, epochs=40)
    m["P7_lr3e3"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet",
                        rank_min_gap=0.01, lr=0.003, epochs=40)
    # G-rows are built on R_mg02 (edge-aware unweighted ranknet min_gap 0.02)
    m["G1"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.02,
                  lambda_global=1.0, global_known_low=False, global_hard_frac=0.0)
    m["G2"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.02,
                  lambda_global=1.0, global_known_low=True, global_hard_frac=0.5)
    m["G1dm"] = _tk(edge_aware=True, weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.02,
                    lambda_global=1.0, global_known_low=False, global_hard_frac=0.0,
                    lambda_dm_contrast=1.0)
    return m


VERSION_ORDER = ["B0", "B1", "B2", "B2e_ctrl", "B3_sim", "B3_simtr", "B3_all", "B4_grouped",
                 "B6_heads", "BEST", "B5_ranknet", "R_tohet", "R_weights", "R_heads",
                 "R_mg00", "R_mg02", "P6_dm05", "P6_dm10", "P7_es", "P7_lr3e3",
                 "G1", "G2", "G1dm"]
