import os
import torch
import argparse
import lightning as L
from datetime import datetime
from data import get_dataloader, compute_audioset_pos_weight, SAMPLE_RATE, HOP_LENGTH, N_MELS
from dataset_config import DATASETS

from models import EchoTrainer

parser = argparse.ArgumentParser(description="Train EchoTrainer")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
parser.add_argument(
    "--lambd", type=float, default=0.02, help="Lambda regularization parameter"
)
parser.add_argument(
    "--num_global_views", type=int, default=2, help="Number of global views (G)"
)
parser.add_argument(
    "--num_local_views", type=int, default=4, help="Number of local views (K)"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=11, help="Number of workers")
parser.add_argument(
    "--num_workers_test", type=int, default=4, help="Number of test workers"
)
parser.add_argument("--num_frames", type=int, default=8, help="Number of video frames")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
)
parser.add_argument(
    "--load_weights_only",
    action="store_true",
    help="Load only model weights from --checkpoint; start fresh optimizer/scheduler/step counter. "
    "Use for continued pretraining where a new LR schedule is desired.",
)
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
)
parser.add_argument(
    "--video_mask", type=float, default=0.80, help="Video masking ratio (0.0-1.0)"
)
parser.add_argument(
    "--audio_mask",
    type=float,
    default=0.5,
    help="Audio masking percentage (0.0-1.0, applied to freq and time masking)",
)
parser.add_argument(
    "--frame_size",
    type=int,
    default=224,
    help="Frame size (height and width) for video frames",
)
parser.add_argument(
    "--vit_size",
    type=str,
    default="base",
    choices=["small", "base", "large"],
    help="ViT model size: 'small' (ViT-S), 'base' (ViT-B), or 'large' (ViT-L)",
)
parser.add_argument(
    "--proj_dim", type=int, default=128, help="Projector output dimension"
)
parser.add_argument("--dataset", type=str, default="vggsound", choices=list(DATASETS.keys()),
                    help="Dataset to use for training")
parser.add_argument("--num_classes", type=int, default=None,
                    help="Number of classes (default: derived from dataset config)")
parser.add_argument("--probe_lr", type=float, default=1e-3)
parser.add_argument("--probe_weight_decay", type=float, default=0.0)
parser.add_argument("--num_eval_clips", type=int, default=4)
parser.add_argument("--run_test", action="store_true")
parser.add_argument("--accumulate_grad", action="store_true")
parser.add_argument(
    "--max_steps", type=int, default=-1,
    help="Cap total optimizer steps (-1 = no cap). Useful for smoke tests.",
)
parser.add_argument(
    "--modality_drop",
    type=float,
    default=0.5,
    help="Probability of dropping one modality in local views (0.0 to disable)",
)
parser.add_argument(
    "--clean_survivor",
    action="store_true",
    help="When a modality is dropped, keep the surviving modality unmasked",
)
parser.add_argument(
    "--cross_modal",
    action="store_true",
    help="Deterministic cross-modal local views: 1 audio-only + 1 video-only, no partial masking",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Enable gradient checkpointing on encoder blocks to reduce memory (~40% activation savings)",
)
parser.add_argument(
    "--num_gpus", type=int, default=1, help="GPUs per node (>1 or num_nodes>1 enables DDP)"
)
parser.add_argument(
    "--num_nodes", type=int, default=1, help="Number of nodes for multi-node DDP"
)
parser.add_argument(
    "--gated_attention",
    type=str,
    default="none",
    choices=["none", "headwise", "elementwise"],
    help="Gated attention variant: 'none' (standard), 'headwise', or 'elementwise'",
)
parser.add_argument(
    "--attentive_probe",
    action="store_true",
    help="Enable attentive probe alongside linear probe",
)
parser.add_argument(
    "--mean_pool",
    action="store_true",
    help="Use mean-pooled patch tokens instead of CLS token for the linear probe",
)
parser.add_argument(
    "--dual_encoder",
    action="store_true",
    help="Use separate audio and video encoders (requires --cross_modal)",
)
parser.add_argument(
    "--augment", action="store_true",
    help="Enable all SSL augmentations with default strengths",
)
parser.add_argument(
    "--color_jitter", type=float, default=0.0,
    help="ColorJitter strength multiplier (0=off, 1.0=default BYOL strength)",
)
parser.add_argument(
    "--gaussian_blur", type=float, default=0.0,
    help="GaussianBlur probability for local views (0=off, 0.5=default)",
)
parser.add_argument(
    "--random_grayscale", type=float, default=0.0,
    help="RandomGrayscale probability for local views (0=off, 0.2=default)",
)
parser.add_argument(
    "--solarize", type=float, default=0.0,
    help="Solarize probability for global views (0=off, 0.2=default)",
)
parser.add_argument(
    "--audio_noise", type=float, default=0.0,
    help="Gaussian noise std for audio spectrograms (0=off, 0.3=default)",
)
parser.add_argument(
    "--audio_gain", type=float, default=0.0,
    help="Random gain range in dB for audio spectrograms (0=off, 5.0=default)",
)
parser.add_argument(
    "--predictive", action="store_true",
    help="Predictive cross-modal JEPA: no global views, predictor bridges audio↔video embeddings",
)
parser.add_argument(
    "--pred_dim", type=int, default=64,
    help="Predictor bottleneck dimension (only used with --predictive)",
)
parser.add_argument(
    "--run_name", type=str, default=None,
    help="W&B run name; falls back to SLURM_JOB_ID when omitted",
)
parser.add_argument(
    "--wandb_id", type=str, default=None,
    help="Stable W&B run id; when set, the run is resumed (resume='allow') so multi-job training shows as a single continuous run",
)
parser.add_argument(
    "--mask_cross_modal", action="store_true",
    help="Apply video tube masking and audio freq/time masking to cross-modal local views",
)
parser.add_argument(
    "--per_modal_loss", action="store_true",
    help="Shared-encoder cross-modal: per-modality center + per-modality SIGReg (mirrors dual-encoder branch)",
)
parser.add_argument(
    "--video_token_crop", action="store_true",
    help="AV-JEPA local views: full-frame video patch-embedded then spatially "
         "cropped by dropping out-of-bbox patch tokens (no pixel crop/resize), "
         "with independent per-modality dropout (p=--modality_drop each, never "
         "both). Standard 2-global / K-local JEPA path; not for use with --cross_modal.",
)
parser.add_argument(
    "--crop_scale", type=float, default=0.4,
    help="Fraction of the spatial patch grid kept by the local token crop "
         "(0.4 -> a 9x9 block of the 14x14 grid at 224px). Only with --video_token_crop.",
)
parser.add_argument(
    "--mm_sigreg", action="store_true",
    help="Use the multimodal SIGReg variant (marginal + joint + independence) "
         "in place of the per-modality LeJEPA SIGReg. Affects cross_modal, "
         "dual_encoder, per_modal_loss, and predictive branches.",
)
parser.add_argument(
    "--mm_sigreg_num_slices", type=int, default=2048,
    help="Number of random projections per MM-SIGReg component.",
)
parser.add_argument(
    "--mm_sigreg_w_marg", type=float, default=1.0,
    help="Weight on the per-modality marginal-isotropy term.",
)
parser.add_argument(
    "--mm_sigreg_w_joint", type=float, default=0.5,
    help="Weight on the joint-space isotropy term (penalises modality gap).",
)
parser.add_argument(
    "--mm_sigreg_w_ind", type=float, default=0.25,
    help="Weight on the paired-difference independence term.",
)
parser.add_argument(
    "--sep_loss", action="store_true",
    help="Add a cross-modal-disagreement separation term: late-ramped batch "
         "repulsion that pushes apart per-sample fused embeddings the encoder "
         "merged but a single modality says differ (the complementary modality "
         "gates which pairs to separate). Canonical shared-encoder loss path only.",
)
parser.add_argument(
    "--sep_weight", type=float, default=0.1,
    help="Maximum weight of the separation term after the ramp.",
)
parser.add_argument(
    "--sep_start_step", type=int, default=5000,
    help="Optimizer step at which the separation ramp begins (0 weight before).",
)
parser.add_argument(
    "--sep_ramp_steps", type=int, default=2000,
    help="Length in steps of the linear ramp from 0 to --sep_weight.",
)
parser.add_argument(
    "--visreg", action="store_true",
    help="Use VISReg (variance + sliced-Wasserstein sketch + centering) as the "
         "embedding-space regularizer in place of SIGReg. Drop-in: keeps the "
         "(1-lambd)*inv + lambd*reg structure, only the reg term changes. VISReg "
         "is O(1) in magnitude (vs SIGReg's ~n_global), so retune --lambd upward.",
)
parser.add_argument(
    "--visreg_num_slices", type=int, default=2048,
    help="Number of random 1-D slices for the VISReg sliced-Wasserstein shape "
         "term (drawn independently per GPU).",
)
parser.add_argument(
    "--visreg_w_scale", type=float, default=1.0,
    help="Weight on the VISReg variance/scale term (1 - std)^2.",
)
parser.add_argument(
    "--visreg_w_shape", type=float, default=1.0,
    help="Weight on the VISReg sliced-Wasserstein shape (sketch) term.",
)
parser.add_argument(
    "--visreg_w_center", type=float, default=1.0,
    help="Weight on the VISReg centering term ||mean||^2.",
)
args = parser.parse_args()

if args.augment:
    if args.color_jitter == 0.0:
        args.color_jitter = 1.0
    if args.gaussian_blur == 0.0:
        args.gaussian_blur = 0.5
    if args.random_grayscale == 0.0:
        args.random_grayscale = 0.2
    if args.solarize == 0.0:
        args.solarize = 0.2
    if args.audio_noise == 0.0:
        args.audio_noise = 0.3
    if args.audio_gain == 0.0:
        args.audio_gain = 5.0

if args.dual_encoder:
    assert args.cross_modal, "--dual_encoder requires --cross_modal"
if args.predictive:
    assert args.cross_modal, "--predictive requires --cross_modal"
    assert args.num_global_views == 0, "--predictive requires --num_global_views 0"
if args.video_token_crop:
    assert not args.cross_modal, "--video_token_crop uses the standard JEPA path; do not combine with --cross_modal"
    assert args.num_global_views > 0, "--video_token_crop requires --num_global_views > 0 (AV global targets)"
if args.sep_loss:
    assert not args.dual_encoder, "--sep_loss only supported on the shared-encoder loss path"
    assert not args.predictive, "--sep_loss not supported with --predictive (no global views)"
    assert not args.per_modal_loss, "--sep_loss uses the canonical loss branch; do not combine with --per_modal_loss"
    assert args.num_global_views > 0, "--sep_loss requires --num_global_views > 0 (fused per-sample centers)"
if args.visreg:
    assert not args.mm_sigreg, "--visreg and --mm_sigreg are mutually exclusive (both replace SIGReg)"

print(f"Using accumulate_grad: {args.accumulate_grad}")

MAX_FREQ_MASK = 128
MAX_TIME_MASK = 801
freq_mask_param = int(MAX_FREQ_MASK * args.audio_mask)
time_mask_param = int(MAX_TIME_MASK * args.audio_mask)

clip_duration = 8
audio_len = SAMPLE_RATE * clip_duration
spec_time = audio_len // HOP_LENGTH + 1

VIT_CONFIGS = {
    "small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "intermediate_size": 4 * 384,
        "num_attention_heads": 6,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 4 * 768,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 4 * 1024,
        "num_attention_heads": 16,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
}

TransformerConfig = VIT_CONFIGS[args.vit_size]
TransformerConfig["gated_attention"] = args.gated_attention
print(
    f"Using ViT-{ {'small': 'S', 'base': 'B', 'large': 'L'}[args.vit_size] } (hidden_size={TransformerConfig['hidden_size']})"
)

AudioConfig = {
    "spectrogram_size": (N_MELS, spec_time),
    "patch_size": (16, 16),
    "patch_stride": (16, 16),
    "num_channels": 1,
}

VideoConfig = {
    "num_frames": args.num_frames,
    "tubelet_size": 2,
    "image_size": args.frame_size,
    "num_channels": 3,
    "patch_size": 16,
}

ds = DATASETS[args.dataset]
print(f"Dataset: {ds.name} ({ds.num_classes} classes, multi_label={ds.multi_label})")
print("Starting data loading...")

train_loader, test_loader, classes = get_dataloader(
    tar_path=ds.train_tar,
    csv_path=ds.train_csv,
    test_tar_path=ds.test_tar,
    test_csv_path=ds.test_csv,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    num_workers_test=args.num_workers_test,
    frame_size=(args.frame_size, args.frame_size),
    num_frames=args.num_frames,
    num_global_views=args.num_global_views,
    num_local_views=args.num_local_views,
    num_eval_clips=args.num_eval_clips,
    train_size=ds.train_size,
    test_size=ds.test_size,
    world_size=args.num_gpus * args.num_nodes,
    video_mask_ratio=args.video_mask,
    freq_mask_param=freq_mask_param,
    time_mask_param=time_mask_param,
    modality_drop_prob=args.modality_drop,
    clean_survivor=args.clean_survivor,
    cross_modal=args.cross_modal,
    mask_cross_modal=args.mask_cross_modal,
    video_token_crop=args.video_token_crop,
    crop_scale=args.crop_scale,
    color_jitter=args.color_jitter,
    gaussian_blur=args.gaussian_blur,
    random_grayscale=args.random_grayscale,
    solarize=args.solarize,
    audio_noise=args.audio_noise,
    audio_gain=args.audio_gain,
    csv_format=ds.csv_format,
    spec_mean=ds.spec_mean,
    spec_std=ds.spec_std,
)
num_classes = args.num_classes if args.num_classes is not None else len(classes)
print(f"Loaded {len(classes)} classes (using {num_classes} for probe).")

pos_weight = None
if ds.multi_label:
    pos_weight = compute_audioset_pos_weight(ds.train_csv, classes)
    print(f"Computed pos_weight: min={pos_weight.min():.1f}, max={pos_weight.max():.1f}, median={pos_weight.median():.1f}")

echo = EchoTrainer(
    a_config=AudioConfig,
    v_config=VideoConfig,
    t_config=TransformerConfig,
    lr=args.lr,
    weight_decay=5e-2,
    lambd=args.lambd,
    num_views=args.num_local_views,
    batch_size=args.batch_size,
    epochs=args.epochs,
    proj_dim=args.proj_dim,
    num_classes=num_classes,
    probe_lr=args.probe_lr,
    probe_weight_decay=args.probe_weight_decay,
    cross_modal=args.cross_modal,
    per_modal_loss=args.per_modal_loss,
    video_token_crop=args.video_token_crop,
    gradient_checkpointing=args.gradient_checkpointing,
    attentive_probe=args.attentive_probe,
    dual_encoder=args.dual_encoder,
    mean_pool=args.mean_pool,
    predictive=args.predictive,
    pred_dim=args.pred_dim,
    multi_label=ds.multi_label,
    total_samples=ds.train_size,
    pos_weight=pos_weight,
    mm_sigreg=args.mm_sigreg,
    mm_sigreg_num_slices=args.mm_sigreg_num_slices,
    mm_sigreg_w_marg=args.mm_sigreg_w_marg,
    mm_sigreg_w_joint=args.mm_sigreg_w_joint,
    mm_sigreg_w_ind=args.mm_sigreg_w_ind,
    sep_loss=args.sep_loss,
    sep_weight=args.sep_weight,
    sep_start_step=args.sep_start_step,
    sep_ramp_steps=args.sep_ramp_steps,
    visreg=args.visreg,
    visreg_num_slices=args.visreg_num_slices,
    visreg_w_scale=args.visreg_w_scale,
    visreg_w_shape=args.visreg_w_shape,
    visreg_w_center=args.visreg_w_center,
)

slurm_id = os.environ.get("SLURM_JOB_ID", "local")
run_name = f"echo_{args.dataset}/{slurm_id}/{datetime.now().strftime('%d-%m-%H:%M:%S')}"
wandb_run_name = args.run_name if args.run_name else slurm_id

checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"{args.checkpoint_dir}/{run_name}",
    filename=f"echo-{args.dataset}" + "-{step}",
    every_n_train_steps=2000,
    save_top_k=-1,
    monitor=None,
)

wandb_logger = L.pytorch.loggers.WandbLogger(
    project=f"echo-{args.dataset}",
    name=wandb_run_name,
    save_dir="runs",
    config={**vars(args), "slurm_job_id": slurm_id},
    id=args.wandb_id,
    resume="allow" if args.wandb_id else None,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

world_size = args.num_gpus * args.num_nodes
strategy = "ddp" if world_size > 1 else "auto"
print(f"Using {args.num_gpus} GPU(s) x {args.num_nodes} node(s) = {world_size} ranks, strategy={strategy}")

trainer = L.Trainer(
    max_epochs=args.epochs,
    max_steps=args.max_steps,
    accelerator="gpu",
    devices=args.num_gpus,
    num_nodes=args.num_nodes,
    strategy=strategy,
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=10,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback],
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=2 if args.accumulate_grad else 1,
)

if args.epochs > 0:
    print(f"Starting Echo ({args.dataset}) training...")
    if args.load_weights_only and args.checkpoint:
        print(f"Loading weights only from: {args.checkpoint} (fresh optimizer/scheduler)")
        ckpt_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = echo.load_state_dict(ckpt_state["state_dict"], strict=False)
        if missing:
            print(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        trainer.fit(echo, train_loader)
    else:
        trainer.fit(echo, train_loader, ckpt_path=args.checkpoint)
    print("Training Complete.")

if args.run_test:
    if args.epochs == 0 and args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")
        echo = EchoTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    trainer.test(echo, dataloaders=test_loader)
