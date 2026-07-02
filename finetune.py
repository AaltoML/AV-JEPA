import os
import torch
import argparse
import lightning as L
from datetime import datetime

from models import EchoTrainer, EchoFineTuner
from data import get_dataloader, compute_audioset_pos_weight, SAMPLE_RATE, HOP_LENGTH, N_MELS
from dataset_config import DATASETS
from utils import HardwareMonitorCallback

parser = argparse.ArgumentParser(description="Finetune a pretrained Echo encoder")
parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to pretrained EchoTrainer checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned checkpoint (for resume/test-only)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="Backbone LR = lr * this scale")
parser.add_argument("--weight_decay", type=float, default=5e-2)
parser.add_argument("--warmup_fraction", type=float, default=0.05)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=11)
parser.add_argument("--num_workers_test", type=int, default=4)
parser.add_argument("--num_frames", type=int, default=8)
parser.add_argument("--frame_size", type=int, default=224)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--freeze_epochs", type=int, default=0, help="Freeze encoder for this many epochs before unfreezing")
parser.add_argument("--vit_size", type=str, default="base", choices=["small", "base", "large"])
parser.add_argument("--num_eval_clips", type=int, default=4)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--gated_attention", type=str, default="none", choices=["none", "headwise", "elementwise"])
parser.add_argument("--attentive_probe", action="store_true")
parser.add_argument("--mean_pool", action="store_true")
parser.add_argument("--dual_encoder", action="store_true")
parser.add_argument("--cross_modal", action="store_true")
parser.add_argument("--accumulate_grad", action="store_true")
parser.add_argument("--run_test", action="store_true")
parser.add_argument("--skip_val", action="store_true",
                    help="Skip validation during fit (still runs --run_test if set).")
parser.add_argument("--dataset", type=str, default="vggsound", choices=list(DATASETS.keys()),
                    help="Dataset to finetune on")
parser.add_argument("--mixup_alpha", type=float, default=0.0,
                    help="Beta(a,a) mixup strength. Multi-label mixes the multi-hot targets; "
                    "single-label combines the CE losses of both endpoint labels. 0.0 disables mixup.")
parser.add_argument("--freq_mask_param", type=int, default=0,
                    help="SpecAugment frequency mask width (mel bins) on the finetune train view. 0 disables.")
parser.add_argument("--time_mask_param", type=int, default=0,
                    help="SpecAugment time mask width (spectrogram frames) on the finetune train view. 0 disables.")
parser.add_argument("--random_resized_crop", action="store_true",
                    help="Use RandomResizedCrop on the finetune train view instead of Resize+CenterCrop.")
parser.add_argument("--rrc_min_scale", type=float, default=0.4,
                    help="Minimum area scale for --random_resized_crop.")
parser.add_argument("--modality_drop_prob", type=float, default=0.0,
                    help="Per-sample probability of zeroing one modality during finetune. "
                    "Half goes to zero-audio, half to zero-video. Trains the classifier on bimodal, "
                    "audio-only, and video-only inputs so single-modality eval is in-distribution.")
parser.add_argument("--no_pos_weight", action="store_true",
                    help="Disable BCE pos_weight for multi-label datasets.")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="Base directory for output finetune checkpoints.")
args = parser.parse_args()

clip_duration = 8
audio_len = SAMPLE_RATE * clip_duration
spec_time = audio_len // HOP_LENGTH + 1

VIT_CONFIGS = {
    "small": {
        "hidden_size": 384, "num_hidden_layers": 12, "intermediate_size": 4 * 384,
        "num_attention_heads": 6, "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1, "qkv_bias": True, "initializer_range": 0.02,
    },
    "base": {
        "hidden_size": 768, "num_hidden_layers": 12, "intermediate_size": 4 * 768,
        "num_attention_heads": 12, "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1, "qkv_bias": True, "initializer_range": 0.02,
    },
    "large": {
        "hidden_size": 1024, "num_hidden_layers": 24, "intermediate_size": 4 * 1024,
        "num_attention_heads": 16, "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1, "qkv_bias": True, "initializer_range": 0.02,
    },
}

TransformerConfig = VIT_CONFIGS[args.vit_size]
TransformerConfig["gated_attention"] = args.gated_attention
print(f"Using ViT-{'S' if args.vit_size == 'small' else 'B'} (hidden_size={TransformerConfig['hidden_size']})")

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

print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
echo_trainer = EchoTrainer.load_from_checkpoint(
    args.pretrained_checkpoint,
    a_config=AudioConfig,
    v_config=VideoConfig,
    t_config=TransformerConfig,
    mm_sigreg=False,
    strict=False,
)
encoder = echo_trainer.encoder
del echo_trainer
torch.cuda.empty_cache()
print("Encoder loaded.")

ds = DATASETS[args.dataset]

print(f"Starting data loading for {ds.name}...")
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
    num_global_views=1,
    num_local_views=1,
    num_eval_clips=args.num_eval_clips,
    train_size=ds.train_size,
    test_size=ds.test_size,
    world_size=args.num_gpus,
    video_mask_ratio=0.0,
    freq_mask_param=args.freq_mask_param,
    time_mask_param=args.time_mask_param,
    spec_aug_global=(args.freq_mask_param > 0 or args.time_mask_param > 0),
    global_rrc_min_scale=args.rrc_min_scale if args.random_resized_crop else 0.0,
    modality_drop_prob=0.0,
    clean_survivor=False,
    cross_modal=False,
    color_jitter=0.0,
    gaussian_blur=0.0,
    random_grayscale=0.0,
    solarize=0.0,
    audio_noise=0.0,
    audio_gain=0.0,
    csv_format=ds.csv_format,
    spec_mean=ds.spec_mean,
    spec_std=ds.spec_std,
)
num_classes = len(classes)
print(f"Loaded {num_classes} classes.")

pos_weight = None
if ds.multi_label and not args.no_pos_weight:
    print("Computing AudioSet pos_weight from train CSV...")
    pos_weight = compute_audioset_pos_weight(ds.train_csv, classes)
    print(f"  pos_weight stats: min={pos_weight.min():.2f}  median={pos_weight.median():.2f}  max={pos_weight.max():.2f}")

model = EchoFineTuner(
    encoder=encoder,
    hidden_size=TransformerConfig["hidden_size"],
    num_classes=num_classes,
    lr=args.lr,
    backbone_lr_scale=args.backbone_lr_scale,
    weight_decay=args.weight_decay,
    warmup_fraction=args.warmup_fraction,
    label_smoothing=args.label_smoothing,
    batch_size=args.batch_size,
    total_samples=ds.train_size,
    epochs=args.epochs,
    freeze_epochs=args.freeze_epochs,
    attentive_probe=args.attentive_probe,
    mean_pool=args.mean_pool,
    num_attention_heads=TransformerConfig["num_attention_heads"],
    multi_label=ds.multi_label,
    pos_weight=pos_weight,
    mixup_alpha=args.mixup_alpha,
    modality_drop_prob=args.modality_drop_prob,
)

slurm_id = os.environ.get("SLURM_JOB_ID", "local")
run_name = f"echo_vgg_finetune/{slurm_id}/{datetime.now().strftime('%d-%m-%H:%M:%S')}"

wandb_logger = L.pytorch.loggers.WandbLogger(
    project="echo-vgg-finetune",
    name=f"{slurm_id}",
    save_dir="runs",
    config=vars(args),
)
tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir="runs", name=slurm_id)

checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"{args.checkpoint_dir}/{run_name}",
    filename="finetune-{step}",
    every_n_train_steps=500,
    save_top_k=-1,
    monitor=None,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

total_devices = args.num_gpus * args.num_nodes
strategy = "ddp" if total_devices > 1 else "auto"
print(f"Using {args.num_nodes} node(s) x {args.num_gpus} GPU(s), strategy={strategy}")

trainer = L.Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=args.num_gpus,
    num_nodes=args.num_nodes,
    strategy=strategy,
    precision="bf16-mixed",
    logger=[wandb_logger, tb_logger],
    log_every_n_steps=10,
    enable_progress_bar=True,
    callbacks=[HardwareMonitorCallback(log_every_n_steps=10), checkpoint_callback],
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=2 if args.accumulate_grad else 1,
)

if args.checkpoint:
    print(f"Loading finetuned checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt
    torch.cuda.empty_cache()

if args.epochs > 0:
    print("Starting finetuning...")
    if args.skip_val:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_dataloaders=test_loader)
    print("Finetuning complete.")

if args.run_test:
    trainer.test(model, dataloaders=test_loader)
