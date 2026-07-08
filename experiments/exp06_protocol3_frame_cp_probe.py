import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cp_factorization import nonnegative_cp_mu  # noqa: E402
from metrics import hpe_li_pck_mmfi, mpjpe_3d  # noqa: E402


SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ACTIONS = [f"A{i:02d}" for i in range(1, 28)]


def scene_for_subject(subject):
    sid = int(subject[1:])
    if 1 <= sid <= 10:
        return "E01"
    if 11 <= sid <= 20:
        return "E02"
    if 21 <= sid <= 30:
        return "E03"
    if 31 <= sid <= 40:
        return "E04"
    raise ValueError(subject)


def hpe_li_protocol3_subject_split():
    train = {}
    val = {}
    seed = 0
    for action in ACTIONS:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(SUBJECTS))
        split = int(np.floor(0.7 * len(SUBJECTS)))
        train_subjects = set(np.array(SUBJECTS)[idx[:split]].tolist())
        val_subjects = set(np.array(SUBJECTS)[idx[split:]].tolist())
        for subject in SUBJECTS:
            if subject in train_subjects:
                train.setdefault(subject, []).append(action)
            if subject in val_subjects:
                val.setdefault(subject, []).append(action)
        seed += 1
    return train, val


def iter_available_frames(data_root, data_form):
    for subject, actions in data_form.items():
        scene = scene_for_subject(subject)
        for action in actions:
            seq_dir = data_root / scene / scene / subject / action
            gt_path = seq_dir / "ground_truth.npy"
            wifi_dir = seq_dir / "wifi-csi"
            if not gt_path.exists() or not wifi_dir.exists():
                continue
            gt = np.load(gt_path).astype(np.float32)
            for idx in range(min(297, len(gt))):
                frame_path = wifi_dir / f"frame{idx + 1:03d}.mat"
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    yield {
                        "scene": scene,
                        "subject": subject,
                        "action": action,
                        "idx": idx,
                        "frame_path": frame_path,
                        "pose": gt[idx],
                    }


def split_val_test(val_items, test_size=0.5, seed=41):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(val_items))
    n_test = int(np.ceil(test_size * len(val_items)))
    test_idx = set(order[:n_test].tolist())
    val = []
    test = []
    for i, item in enumerate(val_items):
        if i in test_idx:
            test.append(item)
        else:
            val.append(item)
    return val, test


def load_hpe_li_frame(frame_path):
    data = loadmat(frame_path)["CSIamp"].astype(np.float32)
    data[np.isinf(data)] = np.nan
    for i in range(data.shape[2]):
        col = data[:, :, i]
        if np.isnan(col).any():
            valid = col[~np.isnan(col)]
            col[np.isnan(col)] = float(valid.mean()) if valid.size else 0.0
    mn = float(np.min(data))
    mx = float(np.max(data))
    if mx > mn:
        data = (data - mn) / (mx - mn)
    else:
        data = np.zeros_like(data)
    return data


def cp_frame_feature(frame, rank, iters):
    a, b, c, _ = nonnegative_cp_mu(frame, rank=rank, iters=iters, seed=rank)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()]).astype(np.float32)


def raw_stats_feature(frame):
    return np.concatenate(
        [
            frame.mean(axis=(1, 2)),
            frame.std(axis=(1, 2)),
            frame.mean(axis=(0, 2)),
            frame.std(axis=(0, 2)),
            frame.mean(axis=(0, 1)),
            frame.std(axis=(0, 1)),
        ]
    ).astype(np.float32)


def featurize(items, rank, iters, max_items=None, progress_every=5000):
    if max_items is not None:
        items = items[:max_items]
    x_cp = []
    x_stats = []
    y = []
    meta = []
    t0 = time.time()
    for n, item in enumerate(items, start=1):
        frame = load_hpe_li_frame(item["frame_path"])
        x_cp.append(cp_frame_feature(frame, rank=rank, iters=iters))
        x_stats.append(raw_stats_feature(frame))
        y.append(item["pose"].ravel())
        meta.append((item["scene"], item["subject"], item["action"], item["idx"]))
        if n % progress_every == 0:
            elapsed = time.time() - t0
            print(f"featurized={n}/{len(items)} elapsed_sec={elapsed:.1f}", flush=True)
    return np.vstack(x_cp), np.vstack(x_stats), np.vstack(y), meta


def evaluate(name, y_true, y_pred):
    return {
        "name": name,
        "mpjpe": mpjpe_3d(y_true, y_pred),
        "pck_50": hpe_li_pck_mmfi(y_true, y_pred, 0.5),
        "pck_40": hpe_li_pck_mmfi(y_true, y_pred, 0.4),
        "pck_30": hpe_li_pck_mmfi(y_true, y_pred, 0.3),
        "pck_20": hpe_li_pck_mmfi(y_true, y_pred, 0.2),
        "pck_10": hpe_li_pck_mmfi(y_true, y_pred, 0.1),
    }


def print_result(res):
    print(
        f"{res['name']}: MPJPE={res['mpjpe']:.3f} "
        f"PCK_50={res['pck_50']:.3f} "
        f"PCK_40={res['pck_40']:.3f} "
        f"PCK_30={res['pck_30']:.3f} "
        f"PCK_20={res['pck_20']:.3f} "
        f"PCK_10={res['pck_10']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MMFi_full" / "extracted")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--cp-iters", type=int, default=35)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "protocol3_frame_cp_probe.npz")
    args = parser.parse_args()

    train_form, val_form = hpe_li_protocol3_subject_split()
    train_items = list(iter_available_frames(args.data_root, train_form))
    val_items_all = list(iter_available_frames(args.data_root, val_form))
    _, test_items = split_val_test(val_items_all)

    print(f"available_train_frames={len(train_items)}")
    print(f"available_valtest_frames={len(val_items_all)}")
    print(f"available_test_frames={len(test_items)}")
    print("expected_full_frames=320760")

    x_cp_train, x_stats_train, y_train, train_meta = featurize(
        train_items, args.rank, args.cp_iters, args.max_train
    )
    x_cp_test, x_stats_test, y_test, test_meta = featurize(
        test_items, args.rank, args.cp_iters, args.max_test
    )

    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    stats_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    cp_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    stats_model.fit(x_stats_train, y_train)
    cp_model.fit(x_cp_train, y_train)

    results = [
        evaluate("mean_pose", y_test, mean_pred),
        evaluate("raw_frame_stats_ridge", y_test, stats_model.predict(x_stats_test)),
        evaluate(f"cp_frame_rank{args.rank}_ridge", y_test, cp_model.predict(x_cp_test)),
    ]
    for res in results:
        print_result(res)

    args.output.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        args.output,
        x_cp_train=x_cp_train,
        x_stats_train=x_stats_train,
        y_train=y_train,
        x_cp_test=x_cp_test,
        x_stats_test=x_stats_test,
        y_test=y_test,
        **{f"{r['name']}_{k}": v for r in results for k, v in r.items() if k != "name"},
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved_npz={args.output}")
    print(f"saved_csv={csv_path}")
    print(f"train_used={len(y_train)} test_used={len(y_test)}")


if __name__ == "__main__":
    main()
