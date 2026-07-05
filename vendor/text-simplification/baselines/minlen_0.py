"""Strong baseline: zero min-length floor (natural EOS, matches simp-length-control tuned)."""


def build_min_length() -> int:
    return 0
