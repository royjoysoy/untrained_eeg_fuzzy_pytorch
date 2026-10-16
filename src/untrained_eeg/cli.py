"""Small helpers shared by the command-line scripts."""


def seed_range(text):
    """'0-29' -> [0, ..., 29], '3' -> [3], '0,5-7' -> [0, 5, 6, 7]."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out
