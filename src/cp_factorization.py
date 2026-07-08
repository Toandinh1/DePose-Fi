import numpy as np


def khatri_rao(a, b):
    return np.einsum("ir,jr->ijr", a, b).reshape(a.shape[0] * b.shape[0], a.shape[1])


def unfold(x, mode):
    return np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1))


def normalize_factors(a, b, c, eps=1e-12):
    rank = a.shape[1]
    weights = np.ones(rank)
    for r in range(rank):
        for factor in (a, b, c):
            n = np.linalg.norm(factor[:, r]) + eps
            factor[:, r] /= n
            weights[r] *= n
    c *= weights[None, :]
    return a, b, c


def cp_reconstruct(a, b, c):
    return np.einsum("lr,dr,tr->ldt", a, b, c)


def nonnegative_cp_mu(x, rank, iters=500, seed=0, eps=1e-9):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)

    rng = np.random.default_rng(seed)
    links, doppler, times = x.shape
    a = rng.random((links, rank)) + 0.1
    b = rng.random((doppler, rank)) + 0.1
    c = rng.random((times, rank)) + 0.1

    x1, x2, x3 = unfold(x, 0), unfold(x, 1), unfold(x, 2)
    history = []
    for i in range(iters):
        kr = khatri_rao(b, c)
        a *= (x1 @ kr) / (a @ ((b.T @ b) * (c.T @ c)) + eps)

        kr = khatri_rao(a, c)
        b *= (x2 @ kr) / (b @ ((a.T @ a) * (c.T @ c)) + eps)

        kr = khatri_rao(a, b)
        c *= (x3 @ kr) / (c @ ((a.T @ a) * (b.T @ b)) + eps)

        a, b, c = normalize_factors(a, b, c)
        if i % 25 == 0 or i == iters - 1:
            err = np.linalg.norm(x - cp_reconstruct(a, b, c)) / (np.linalg.norm(x) + eps)
            history.append(err)
    return a, b, c, np.asarray(history)
