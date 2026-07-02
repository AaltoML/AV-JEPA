import os
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from models import EchoTrainer
from data import VideoAudioPipeline, SAMPLE_RATE, HOP_LENGTH, N_MELS
from dataset_config import DATASETS


def _get_token_layout(model):
    """Return token layout metadata from a loaded EchoTrainer."""
    emb = model.encoder.embedding
    num_temporal = emb.num_temporal_patches
    grid_h = emb.video_patch_embed.image_size // emb.video_patch_embed.patch_size
    grid_w = grid_h
    num_video_tokens = num_temporal * grid_h * grid_w
    a_feat_h = emb.audio_patch_embed.num_patches_h
    a_feat_w = emb.audio_patch_embed.num_patches_w
    tubelet_size = emb.video_patch_embed.tubelet_size
    return dict(
        num_temporal=num_temporal,
        grid_h=grid_h, grid_w=grid_w,
        num_video_tokens=num_video_tokens,
        a_feat_h=a_feat_h, a_feat_w=a_feat_w,
        tubelet_size=tubelet_size,
    )


VIT_CONFIGS = {
    "small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "intermediate_size": 4 * 384,
        "num_attention_heads": 6,
        "attention_probs_dropout_prob": 0.0,
        "hidden_dropout_prob": 0.0,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 4 * 768,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.0,
        "hidden_dropout_prob": 0.0,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 4 * 1024,
        "num_attention_heads": 16,
        "attention_probs_dropout_prob": 0.0,
        "hidden_dropout_prob": 0.0,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Echo A↔V retrieval evaluation + t-SNE")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="vggsound", choices=list(DATASETS.keys()))
    p.add_argument("--vit_size", type=str, default="base", choices=["small", "base", "large"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_eval_clips", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--samples_per_class", type=int, default=5)
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--full_eval", action="store_true",
                   help="Use the full eval set instead of the per-class subset")
    p.add_argument("--use_cls", action="store_true",
                   help="Use the CLS token instead of mean-pooled per-modality patches")
    p.add_argument("--use_projector", action="store_true",
                   help="Apply the projector head before computing similarity")
    p.add_argument("--center", action="store_true",
                   help="Subtract the per-modality mean embedding before L2-norm "
                        "(removes the audio/video modality gap; transductive over the eval set)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap on samples processed (debug knob)")
    p.add_argument("--no_tsne", action="store_true")
    p.add_argument("--tsne_samples", type=int, default=3000)
    p.add_argument("--tsne_top_classes", type=int, default=20)
    p.add_argument("--draw_pairs", action="store_true",
                   help="Draw a t-SNE figure with pair-line overlays for ~80 pairs")
    p.add_argument("--output_dir", type=str, default="./runs/retrieval")
    p.add_argument("--output_path", type=str, default=None)
    return p.parse_args()


def build_subset_keys(ds, samples_per_class, seed, csv_path=None):
    """Read a CSV and pick K samples per class with a fixed seed.

    Defaults to the eval CSV; pass ``csv_path`` (e.g. ``ds.train_csv``) to build a
    balanced subset of another split.
    """
    rng = random.Random(seed)
    csv_path = csv_path or ds.test_csv

    if ds.csv_format == "vggsound":
        df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
        df["key"] = df["filename"].astype(str).apply(lambda x: os.path.splitext(x)[0])
        by_class = defaultdict(list)
        for _, row in df.iterrows():
            by_class[row["label"]].append(row["key"])
    else:
        rows = []
        all_class_counts = defaultdict(int)
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(", ", 3)
                if len(parts) < 4:
                    continue
                ytid = parts[0]
                start_ms = round(float(parts[1]) * 1000)
                end_ms = round(float(parts[2]) * 1000)
                label_ids = parts[3].strip('"').split(",")
                key = f"{ytid}_{start_ms}_{end_ms}"
                rows.append((key, label_ids))
                for lid in label_ids:
                    all_class_counts[lid] += 1
        by_class = defaultdict(list)
        for key, label_ids in rows:
            if not label_ids:
                continue
            rarest = min(label_ids, key=lambda l: all_class_counts[l])
            by_class[rarest].append(key)

    chosen = set()
    for cls in sorted(by_class.keys()):
        keys = by_class[cls]
        rng.shuffle(keys)
        for k in keys[:samples_per_class]:
            chosen.add(k)
    return chosen


def build_loader(args, ds, subset_keys=None, csv_override=None, tar_override=None):
    csv_path = csv_override or ds.test_csv
    tar_path = tar_override or ds.test_tar
    pipeline = VideoAudioPipeline(
        csv_path,
        is_train=False,
        debug=True,
        frame_size=(args.frame_size, args.frame_size),
        num_frames=args.num_frames,
        num_eval_clips=args.num_eval_clips,
        csv_format=ds.csv_format,
        spec_mean=ds.spec_mean,
        spec_std=ds.spec_std,
    )

    if subset_keys is not None:
        before = len(pipeline.labels_map)
        pipeline.labels_map = {
            k: v for k, v in pipeline.labels_map.items() if k in subset_keys
        }
        print(f"  Subset filter: {len(pipeline.labels_map)} of {before} eval samples retained "
              f"(subset has {len(subset_keys)} keys)")

    dataset = wds.WebDataset(
        tar_path,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        empty_check=False,
    )
    dataset = (
        dataset.select(pipeline.has_label)
        .map(pipeline.process)
        .select(lambda x: x is not None)
        .batched(args.batch_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=True,
    )
    return loader, pipeline.classes


def extract_embeddings(model, loader, args, layout, device):
    audio_embs, video_embs, labels = [], [], []
    num_video_tokens = layout["num_video_tokens"]
    n = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            audio = batch["spectrogram"].to(device, non_blocking=True)
            label = batch["label"]

            B, N = video.shape[:2]
            video_flat = video.view(B * N, *video.shape[2:])
            audio_flat = audio.view(B * N, *audio.shape[2:])

            zero_v = torch.zeros_like(video_flat)
            zero_a = torch.zeros_like(audio_flat)

            if args.use_cls:
                a = model.encoder(zero_v, audio_flat)
                v = model.encoder(video_flat, zero_a)
            else:
                _, audio_patches = model.encoder(zero_v, audio_flat, return_patches=True)
                _, video_patches = model.encoder(video_flat, zero_a, return_patches=True)
                a = audio_patches[:, num_video_tokens:].mean(dim=1)
                v = video_patches[:, :num_video_tokens].mean(dim=1)

            if args.use_projector:
                a = model.projector(a)
                v = model.projector(v)

            a = a.view(B, N, -1).mean(dim=1)
            v = v.view(B, N, -1).mean(dim=1)

            audio_embs.append(a.float().cpu())
            video_embs.append(v.float().cpu())
            if torch.is_tensor(label):
                labels.append(label.cpu())
            else:
                labels.append(torch.as_tensor(label))

            n += B
            if n % 256 < B or n < 64:
                print(f"  Processed {n} samples")
            if args.max_samples is not None and n >= args.max_samples:
                break

    A = torch.cat(audio_embs)
    V = torch.cat(video_embs)
    L = torch.cat(labels) if labels[0].dim() > 0 else torch.stack(labels).flatten()
    if args.max_samples is not None:
        A = A[: args.max_samples]
        V = V[: args.max_samples]
        L = L[: args.max_samples]
    return A, V, L


def compute_retrieval_metrics(A, V, ks=(1, 5, 10), center=False):
    if center:
        A = A - A.mean(dim=0, keepdim=True)
        V = V - V.mean(dim=0, keepdim=True)
    A = F.normalize(A, dim=-1)
    V = F.normalize(V, dim=-1)
    S = A @ V.t()
    N = S.shape[0]

    def _ranks(sim):
        diag = sim.diag().unsqueeze(1)
        return (sim > diag).sum(dim=1) + 1

    a2v = _ranks(S).float()
    v2a = _ranks(S.t()).float()

    out = {"N": N}
    for direction, r in [("a2v", a2v), ("v2a", v2a)]:
        for k in ks:
            out[f"{direction}_R@{k}"] = (r <= k).float().mean().item() * 100
        out[f"{direction}_mean_rank"] = r.mean().item()
        out[f"{direction}_median_rank"] = float(r.median().item())
    return out


def format_metrics(m):
    lines = [f"N = {m['N']}"]
    for d in ["a2v", "v2a"]:
        line = f"  {d.upper():3s}: " + "  ".join(
            f"R@{k}={m[f'{d}_R@{k}']:5.2f}" for k in [1, 5, 10]
        )
        line += f"   mean={m[f'{d}_mean_rank']:6.1f}   median={m[f'{d}_median_rank']:5.0f}"
        lines.append(line)
    return "\n".join(lines)


def plot_tsne(A, V, labels, args, out_dir, dataset_name, ckpt_id, multi_label):
    if args.no_tsne:
        return

    n_total = A.shape[0]
    n = min(args.tsne_samples, n_total)
    rng = np.random.default_rng(args.subset_seed)
    idx = rng.choice(n_total, size=n, replace=False)

    A_sub = F.normalize(A[idx], dim=-1).numpy()
    V_sub = F.normalize(V[idx], dim=-1).numpy()

    print(f"Running t-SNE on {2 * n} points...")
    X = np.concatenate([A_sub, V_sub], axis=0)
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (2 * n - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=args.subset_seed,
    )
    Y = tsne.fit_transform(X)
    Y_a, Y_v = Y[:n], Y[n:]

    L_sub_export = labels[idx]
    if multi_label and L_sub_export.dim() == 2:
        cls_export = L_sub_export.argmax(dim=1).numpy()
    else:
        cls_export = L_sub_export.numpy() if torch.is_tensor(L_sub_export) else np.asarray(L_sub_export)
    csv_audio = os.path.join(out_dir, f"{dataset_name}_{ckpt_id}_tsne_audio.csv")
    csv_video = os.path.join(out_dir, f"{dataset_name}_{ckpt_id}_tsne_video.csv")
    with open(csv_audio, "w") as f:
        f.write("x y class\n")
        for i in range(n):
            f.write(f"{Y_a[i, 0]:.4f} {Y_a[i, 1]:.4f} {int(cls_export[i])}\n")
    with open(csv_video, "w") as f:
        f.write("x y class\n")
        for i in range(n):
            f.write(f"{Y_v[i, 0]:.4f} {Y_v[i, 1]:.4f} {int(cls_export[i])}\n")
    print(f"  Saved {csv_audio}")
    print(f"  Saved {csv_video}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Y_a[:, 0], Y_a[:, 1], c="#f4a142", s=4, alpha=0.6, label="Audio")
    ax.scatter(Y_v[:, 0], Y_v[:, 1], c="#4285f4", s=4, alpha=0.6, label="Video")
    ax.legend(loc="upper right")
    ax.set_title(f"Joint embedding t-SNE — {dataset_name}\n{ckpt_id}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    p1 = os.path.join(out_dir, f"{dataset_name}_{ckpt_id}_tsne_modality.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p1}")

    L_sub = labels[idx]
    if multi_label and L_sub.dim() == 2:
        class_ids = L_sub.argmax(dim=1).numpy()
    else:
        class_ids = L_sub.numpy() if torch.is_tensor(L_sub) else np.asarray(L_sub)

    unique, counts = np.unique(class_ids, return_counts=True)
    top_k = min(args.tsne_top_classes, len(unique))
    top_classes = unique[np.argsort(-counts)[:top_k]]
    is_top = np.isin(class_ids, top_classes)

    cmap = plt.get_cmap("tab20", max(top_k, 1))
    color_map = {c: cmap(i) for i, c in enumerate(top_classes)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, Y_mod, mod_name in [(axes[0], Y_a, "Audio"), (axes[1], Y_v, "Video")]:
        ax.scatter(Y_mod[~is_top, 0], Y_mod[~is_top, 1],
                   c="lightgrey", s=2, alpha=0.3)
        for c in top_classes:
            mask = (class_ids == c) & is_top
            if mask.any():
                ax.scatter(Y_mod[mask, 0], Y_mod[mask, 1],
                           c=[color_map[c]], s=10, alpha=0.75)
        ax.set_title(f"{mod_name} embeddings — top {top_k} classes")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"t-SNE class structure — {dataset_name}", fontsize=13)
    p2 = os.path.join(out_dir, f"{dataset_name}_{ckpt_id}_tsne_class.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p2}")

    if args.draw_pairs:
        n_pairs = min(80, n)
        sub_idx = np.random.default_rng(args.subset_seed).choice(n, size=n_pairs, replace=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in sub_idx:
            ax.plot([Y_a[i, 0], Y_v[i, 0]], [Y_a[i, 1], Y_v[i, 1]],
                    color="grey", alpha=0.25, linewidth=0.5)
        ax.scatter(Y_a[sub_idx, 0], Y_a[sub_idx, 1],
                   c="#f4a142", s=14, alpha=0.85, label="Audio")
        ax.scatter(Y_v[sub_idx, 0], Y_v[sub_idx, 1],
                   c="#4285f4", s=14, alpha=0.85, label="Video")
        ax.legend()
        ax.set_title(f"A↔V pair alignment — {n_pairs} pairs ({dataset_name})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        p3 = os.path.join(out_dir, f"{dataset_name}_{ckpt_id}_tsne_pairs.png")
        plt.tight_layout()
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {p3}")


def derive_ckpt_id(checkpoint_path):
    p = Path(checkpoint_path)
    return f"{p.parent.parent.name}_{p.stem.replace('=', '')}"


def load_model(checkpoint_path, vit_size, num_frames, frame_size, device):
    """Load a frozen EchoTrainer checkpoint and return (model, token_layout)."""
    t_config = VIT_CONFIGS[vit_size]
    spec_time = SAMPLE_RATE * 8 // HOP_LENGTH + 1
    a_config = {
        "spectrogram_size": (N_MELS, spec_time),
        "patch_size": (16, 16),
        "patch_stride": (16, 16),
        "num_channels": 1,
    }
    v_config = {
        "num_frames": num_frames,
        "tubelet_size": 2,
        "image_size": frame_size,
        "num_channels": 3,
        "patch_size": 16,
    }
    print(f"Loading checkpoint: {checkpoint_path}")
    model = EchoTrainer.load_from_checkpoint(
        checkpoint_path,
        a_config=a_config,
        v_config=v_config,
        t_config=t_config,
        strict=False,
    )
    model = model.to(device).eval()
    layout = _get_token_layout(model)
    print(f"  Token layout: video_tokens={layout['num_video_tokens']}, "
          f"audio_tokens={layout['a_feat_h'] * layout['a_feat_w']}")
    return model, layout


def main():
    args = parse_args()
    ds = DATASETS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, layout = load_model(
        args.checkpoint_path, args.vit_size, args.num_frames, args.frame_size, device
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.full_eval:
        subset_keys = None
        print(f"Using full {args.dataset} eval set ({ds.test_size} samples)")
    else:
        cache_path = os.path.join(
            out_dir,
            f"{args.dataset}_{args.samples_per_class}perclass_seed{args.subset_seed}_keys.txt",
        )
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                subset_keys = {line.strip() for line in f if line.strip()}
            print(f"Loaded cached subset: {len(subset_keys)} keys ← {cache_path}")
        else:
            print(f"Building {args.samples_per_class}-per-class subset (seed={args.subset_seed})...")
            subset_keys = build_subset_keys(ds, args.samples_per_class, args.subset_seed)
            with open(cache_path, "w") as f:
                for k in sorted(subset_keys):
                    f.write(k + "\n")
            print(f"  Wrote {len(subset_keys)} keys → {cache_path}")

    print("Building eval loader...")
    loader, _ = build_loader(args, ds, subset_keys=subset_keys)

    print("Extracting embeddings...")
    A, V, labels = extract_embeddings(model, loader, args, layout, device)
    print(f"  Got {A.shape[0]} samples, dim={A.shape[1]}")

    metrics = compute_retrieval_metrics(A, V, center=args.center)
    print("\n=== Retrieval ===")
    if args.center:
        print("(per-modality centering enabled)")
    print(format_metrics(metrics))

    ckpt_id = derive_ckpt_id(args.checkpoint_path)
    output_path = args.output_path or os.path.join(
        out_dir, f"{args.dataset}_{ckpt_id}.json"
    )
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to {output_path}")

    plot_tsne(A, V, labels, args, out_dir, args.dataset, ckpt_id,
              multi_label=ds.multi_label)

    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        try:
            import wandb
            wandb.init(project=f"echo-{args.dataset}", id=run_id, resume="allow")
            wandb.log({f"retrieval/{k}": v for k, v in metrics.items()
                       if isinstance(v, (int, float))})
            print(f"Logged to W&B run {run_id}")
        except Exception as e:
            print(f"W&B logging skipped: {e}")


if __name__ == "__main__":
    main()
