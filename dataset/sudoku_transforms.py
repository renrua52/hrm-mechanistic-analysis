def sudoku_cyclic_shift(x, perm: int):
    mask = (x >= 2) & (x <= 10)

    idx = x[mask] - 2

    idx_shifted = (idx + perm) % 9

    val_shifted = idx_shifted + 2

    out = x.clone()
    out[mask] = val_shifted.to(out.dtype)
    return out