from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


L = 3
S = 114
P = 10
R = 4
D_CP = R * (L + S + P)
OUT = 17 * 3
HIDDEN = 64


def linear_complexity(in_dim, out_dim):
    params = in_dim * out_dim + out_dim
    macs = in_dim * out_dim
    flops = 2 * macs
    return params, macs, flops


def mlp_complexity(in_dim, hidden, out_dim):
    params = in_dim * hidden + hidden + hidden * out_dim + out_dim
    macs = in_dim * hidden + hidden * out_dim
    flops = 2 * macs + hidden  # ReLU comparisons/ops as one op each.
    return params, macs, flops


def cp_iteration_macs(l, s, p, r):
    # Main cost: three MTTKRP-style products for A, B, C updates.
    # A update: (L x SP) @ (SP x R)
    # B update: (S x LP) @ (LP x R)
    # C update: (P x LS) @ (LS x R)
    a_mttkrp = l * s * p * r
    b_mttkrp = s * l * p * r
    c_mttkrp = p * l * s * r

    # Gram products and small R x R products.
    gram = (l + s + p) * (r * r)
    denom = (l + s + p) * (r * r)
    elem = 3 * (l + s + p) * r
    return a_mttkrp + b_mttkrp + c_mttkrp + gram + denom + elem


def fmt_int(x):
    return f"{int(round(x)):,}"


def main():
    ridge_params, ridge_macs, ridge_flops = linear_complexity(D_CP, OUT)
    mlp_params, mlp_macs, mlp_flops = mlp_complexity(D_CP, HIDDEN, OUT)
    cp_macs_iter = cp_iteration_macs(L, S, P, R)

    rows = []
    for iters in [5, 10, 35]:
        cp_macs = cp_macs_iter * iters
        cp_flops = 2 * cp_macs
        rows.append(
            {
                "model": f"CP{iters}+Ridge",
                "feature_dim": D_CP,
                "params": ridge_params,
                "cp_iters": iters,
                "cp_macs": cp_macs,
                "regressor_macs": ridge_macs,
                "total_macs": cp_macs + ridge_macs,
                "total_flops": cp_flops + ridge_flops,
            }
        )
        rows.append(
            {
                "model": f"CP{iters}+MLP64",
                "feature_dim": D_CP,
                "params": mlp_params,
                "cp_iters": iters,
                "cp_macs": cp_macs,
                "regressor_macs": mlp_macs,
                "total_macs": cp_macs + mlp_macs,
                "total_flops": cp_flops + mlp_flops,
            }
        )

    out = ROOT / "outputs" / "flops_params_estimate.csv"
    out.parent.mkdir(exist_ok=True)
    keys = list(rows[0])
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    print(f"CSI frame shape: {L} x {S} x {P} = {L*S*P} raw values")
    print(f"CP rank: {R}")
    print(f"CP feature dim: {D_CP}")
    print()
    print("Downstream regressor only:")
    print(f"  Ridge params={fmt_int(ridge_params)} MACs={fmt_int(ridge_macs)} FLOPs={fmt_int(ridge_flops)}")
    print(f"  MLP64 params={fmt_int(mlp_params)} MACs={fmt_int(mlp_macs)} FLOPs={fmt_int(mlp_flops)}")
    print()
    print(f"Approx CP update cost per iteration: MACs={fmt_int(cp_macs_iter)} FLOPs={fmt_int(2*cp_macs_iter)}")
    print()
    for row in rows:
        print(
            f"{row['model']}: params={fmt_int(row['params'])} "
            f"total_MACs={fmt_int(row['total_macs'])} "
            f"total_FLOPs={fmt_int(row['total_flops'])}"
        )
    print(f"saved_csv={out}")


if __name__ == "__main__":
    main()
