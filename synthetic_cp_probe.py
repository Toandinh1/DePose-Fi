import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def khatri_rao(a, b):
    # Column-wise Kronecker product.
    return np.einsum("ir,jr->ijr", a, b).reshape(a.shape[0] * b.shape[0], a.shape[1])


def unfold(x, mode):
    return np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1))


def normalize_factors(a, b, c, eps=1e-12):
    r = a.shape[1]
    weights = np.ones(r)
    for m in range(r):
        for factor in (a, b, c):
            n = np.linalg.norm(factor[:, m]) + eps
            factor[:, m] /= n
            weights[m] *= n
    c *= weights[None, :]
    return a, b, c


def cp_reconstruct(a, b, c):
    return np.einsum("lr,dr,tr->ldt", a, b, c)


def nonnegative_cp_mu(x, rank, iters=600, seed=0, eps=1e-9):
    rng = np.random.default_rng(seed)
    l, d, t = x.shape
    a = rng.random((l, rank)) + 0.1
    b = rng.random((d, rank)) + 0.1
    c = rng.random((t, rank)) + 0.1

    x1, x2, x3 = unfold(x, 0), unfold(x, 1), unfold(x, 2)
    errors = []
    for i in range(iters):
        kr = khatri_rao(b, c)
        denom = a @ ((b.T @ b) * (c.T @ c)) + eps
        a *= (x1 @ kr) / denom

        kr = khatri_rao(a, c)
        denom = b @ ((a.T @ a) * (c.T @ c)) + eps
        b *= (x2 @ kr) / denom

        kr = khatri_rao(a, b)
        denom = c @ ((a.T @ a) * (b.T @ b)) + eps
        c *= (x3 @ kr) / denom

        a, b, c = normalize_factors(a, b, c)
        if i % 25 == 0 or i == iters - 1:
            err = np.linalg.norm(x - cp_reconstruct(a, b, c)) / np.linalg.norm(x)
            errors.append(err)
    return a, b, c, np.array(errors)


def gaussian_grid(n, center, width):
    grid = np.arange(n)
    return np.exp(-0.5 * ((grid - center) / width) ** 2)


def make_synthetic_tensor(l=8, d=32, t=80, rank=4, noise=0.05, seed=7):
    rng = np.random.default_rng(seed)
    a = np.zeros((l, rank))
    b = np.zeros((d, rank))
    c = np.zeros((t, rank))

    for r in range(rank):
        active_links = rng.choice(l, size=2, replace=False)
        a[active_links, r] = rng.uniform(0.6, 1.2, size=2)
        a[:, r] += 0.05 * rng.random(l)

        center = rng.uniform(5, d - 6)
        width = rng.uniform(1.5, 4.0)
        b[:, r] = gaussian_grid(d, center, width)

        start = int(rng.uniform(0, t * 0.65))
        duration = int(rng.uniform(t * 0.15, t * 0.35))
        end = min(t, start + duration)
        window = np.hanning(max(4, end - start))
        c[start:end, r] = window[: end - start]
        c[:, r] += 0.02 * rng.random(t)

    x_clean = cp_reconstruct(a, b, c)
    x = x_clean + noise * np.max(x_clean) * rng.random(x_clean.shape)
    return x, a, b, c


def column_corr(x, y):
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)
    y = y / (np.linalg.norm(y, axis=0, keepdims=True) + 1e-12)
    return np.abs(x.T @ y)


def evaluate(true_factors, est_factors):
    scores = []
    names = ["link", "doppler", "time"]
    for name, true, est in zip(names, true_factors, est_factors):
        corr = column_corr(true, est)
        row, col = linear_sum_assignment(-corr)
        scores.append((name, float(corr[row, col].mean()), corr))
    return scores


def main():
    x, a_true, b_true, c_true = make_synthetic_tensor()
    a, b, c, errors = nonnegative_cp_mu(x, rank=4)
    scores = evaluate((a_true, b_true, c_true), (a, b, c))

    print("Synthetic link-Doppler-time CP sanity check")
    print(f"tensor_shape={x.shape}")
    print(f"relative_reconstruction_error={np.linalg.norm(x - cp_reconstruct(a,b,c)) / np.linalg.norm(x):.4f}")
    for name, score, _ in scores:
        print(f"mean_matched_{name}_correlation={score:.3f}")

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].imshow(a_true, aspect="auto")
    axes[0, 0].set_title("True link factors")
    axes[0, 1].imshow(a, aspect="auto")
    axes[0, 1].set_title("Estimated link factors")
    axes[1, 0].imshow(b_true, aspect="auto")
    axes[1, 0].set_title("True Doppler factors")
    axes[1, 1].imshow(b, aspect="auto")
    axes[1, 1].set_title("Estimated Doppler factors")
    axes[2, 0].plot(c_true)
    axes[2, 0].set_title("True temporal activations")
    axes[2, 1].plot(c)
    axes[2, 1].set_title("Estimated temporal activations")
    fig.savefig("synthetic_cp_probe.png", dpi=180)
    print("saved=synthetic_cp_probe.png")


if __name__ == "__main__":
    main()
