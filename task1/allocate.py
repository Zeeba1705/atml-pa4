def mean_sparsity(raw, sizes, t, s_min, s_max):
    total= sum(sizes)

    return sum(min(max(r + t, s_min), s_max) * n for r, n in zip(raw, sizes)) / total


def rescale_sparsities(raw, sizes, target, s_min, s_max):
    #additive shift t, clipped to [s_min, s_max]; the mean is monotone in t so bisection finds it
    assert s_min <= target <= s_max

    lo= s_min - max(raw)
    hi= s_max - min(raw)

    for _ in range(200):
        mid= (lo + hi) / 2

        if mean_sparsity(raw, sizes, mid, s_min, s_max) < target:
            lo= mid
        else:
            hi= mid

    t= (lo + hi) / 2

    return [min(max(r + t, s_min), s_max) for r in raw]


def integer_zero_counts(s, sizes, target_zeros):
    #floor, then largest-remainder so the total is exact (decision 1.6)
    exact= [si * n for si, n in zip(s, sizes)]
    zeros= [int(e) for e in exact]

    missing= target_zeros - sum(zeros)
    assert 0 <= missing <= len(zeros), missing

    order= sorted(range(len(zeros)), key=lambda i: exact[i] - zeros[i], reverse=True)

    for i in order[:missing]:
        zeros[i]+= 1

    assert sum(zeros) == target_zeros

    return zeros


def row_counts(n_zeros, n_rows):
    base= n_zeros // n_rows
    extra= n_zeros % n_rows

    return [base + 1 if i < extra else base for i in range(n_rows)]
