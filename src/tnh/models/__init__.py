"""Model wrappers. The default move model is `default_move_model()`."""

DEFAULT_MOVE_MODEL = "maia3-79m-lc0pol-trees+result-cheap"


def default_move_model(**kwargs):
    """The standard move model: `tnh.eval.result.PolicyTreesValued`, the moves of
    `tnh.models.combo.PolicyTrees` (Maia-3 79M at +0 and +600, Lc0 BT4 policy, boosted trees)
    with `wdl` from the cheap result model on the same three network calls. The more accurate
    and about 4 times slower move model is `tnh.models.combo.Combo(prune="p0.5", trees=True)`."""
    from tnh.eval.result import PolicyTreesValued

    return PolicyTreesValued(**kwargs)
