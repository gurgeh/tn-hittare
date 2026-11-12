"""Model wrappers. The default move model is `default_move_model()`."""

DEFAULT_MOVE_MODEL = "maia3-79m-lc0pol-trees"


def default_move_model(**kwargs):
    """The standard move model: `tnh.models.combo.PolicyTrees` (Maia-3 79M at +0 and +600,
    Lc0 BT4 policy, boosted trees; `wdl` from Lc0's value head). The more accurate and about
    4 times slower option is `tnh.models.combo.Combo(prune="p0.5", trees=True)`."""
    from tnh.models.combo import PolicyTrees

    return PolicyTrees(**kwargs)
