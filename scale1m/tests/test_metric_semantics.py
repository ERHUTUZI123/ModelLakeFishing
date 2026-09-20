import pytest

from scale1m import metric_semantics as M


@pytest.mark.parametrize("raw,want", [
    ("Pass@1", "pass_at_1"),
    ("cosine_ndcg@10", "cosine_ndcg_at_10"),
    ("NDCG at 10", "ndcg_at_10"),
    ("f1-macro", "f1_macro"),
    ("  Accuracy  ", "accuracy"),
    ("word.error.rate", "word_error_rate"),
])
def test_name_normalisation(raw, want):
    assert M.normalize_name(raw) == want


def test_cutoff_and_similarity_prefix_are_stripped_for_lookup_only():
    assert M.base_name("cosine_ndcg_at_10") == "ndcg"
    assert M.base_name("euclidean_spearman") == "spearman"
    assert M.normalize_name("cosine_ndcg@10") == "cosine_ndcg_at_10"


@pytest.mark.parametrize("raw,cls", [
    ("accuracy", "higher"), ("f1", "higher"), ("ndcg_at_10", "higher"),
    ("cosine_map@100", "higher"), ("euclidean_pearson", "higher"),
    ("main_score", "higher"), ("bleu", "higher"),
    ("wer", "lower"), ("cer", "lower"), ("eval_loss", "lower"),
    ("perplexity", "lower"), ("rmse", "lower"),
    ("mean_reward", "reward"), ("episode_reward", "reward"),
    ("nauc_map_at_1000_diff1", "unknown"), ("std_reward", "unknown"),
    ("my_custom_score", "unknown"), ("score", "unknown"), ("", "unknown"),
])
def test_direction_classes(raw, cls):
    assert M.classify(raw)[2] == cls


def test_an_unknown_name_is_never_guessed():
    for junk in ("qualityIndexV2", "北京指标", "metric_7", "xyz@5"):
        assert M.classify(junk)[2] == "unknown"
        assert not M.in_gold(M.classify(junk)[2])


def test_only_direction_known_classes_may_define_gold():
    assert M.in_gold("higher") and M.in_gold("lower")
    assert not M.in_gold("reward") and not M.in_gold("unknown")


def test_orientation_flips_lower_is_better_after_normalisation():
    assert M.orient(0.9, "higher") == pytest.approx(0.9)
    assert M.orient(0.9, "lower") == pytest.approx(0.1)
    assert M.orient(0.25, "reward") == pytest.approx(0.25)
