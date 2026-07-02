# AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning

Official code for the paper **"AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning"**
(ICML 2026 Workshop on Machine Learning for Audio).

**Project page: [jepa.benjaminhr.com](https://jepa.benjaminhr.com)**

_Benjamin Robson, Santeri Mentu, Wenshuai Zhao, Arno Solin_
ELLIS Institute Finland and Department of Computer Science, Aalto University

> Note: the model is referred to as `Echo` in the code (`Echo`, `DualEcho`,
> `EchoTrainer` in `models.py`). This is the same model as AV-JEPA in the paper.

## Repository layout

| File                | Purpose                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `train.py`          | Self-supervised pretraining entry point                                         |
| `finetune.py`       | End-to-end fine-tuning / frozen linear & attentive probes                       |
| `retrieval.py`      | Cross-modal audio-video retrieval (R@1/5/10)                                    |
| `models.py`         | `Echo` encoder, `EchoTrainer` (LightningModule), SIGReg, projector, probes      |
| `fusion.py`         | Early-fusion audio/video patch embeddings                                       |
| `transformer.py`    | ViT encoder blocks                                                              |
| `data.py`           | WebDataset video/audio pipeline, mel spectrograms, masking, multi-view sampling |
| `dataset_config.py` | Dataset registry (tar patterns, CSVs, mel statistics)                           |
| `jobs/`             | Example SLURM job scripts                                                       |

## Setup

```bash
mamba env create --file environment.yml
source activate av-jepa
```

Key dependencies: PyTorch 2.9.1 (CUDA 12.9), PyTorch Lightning, torchcodec,
torchaudio, webdataset, wandb.

## Data

Training reads WebDataset tar shards where each sample is an `.mp4` clip
(video plus audio track), with labels looked up from the dataset CSV by sample
key. Point the code at your data with environment variables (defaults are
`./data/VGGSound` and `./data/AudioSet`):

```bash
export VGGSOUND_DIR=/path/to/VGGSound
export AUDIOSET_DIR=/path/to/AudioSet
```

Expected layout (shard counts and patterns are configured in
`dataset_config.py` and can be edited to match your sharding):

```
$VGGSOUND_DIR/
  train_tars/vggsound_train_{00..71}.tar
  test_tars/vggsound_test_{00..03}.tar
  train.csv                       # VGGSound label CSV
  test.csv
$AUDIOSET_DIR/
  shards/data_{000..453}.tar
  unbalanced_train_segments.csv   # AudioSet segment CSVs
  balanced_train_segments.csv
  eval_segments.csv
```

Available `--dataset` choices: `vggsound`, `vggsound_256`, `audioset`,
`audioset_256`, `audioset_20k`. The `_256` variants expect 256px-resized
videos. Per-dataset mel-spectrogram normalization statistics are precomputed
in `dataset_config.py`.

## Pretraining

Multi-GPU pretraining on VGGSound (paper configuration):

```bash
cd jobs
sbatch pretrain-vggsound-8gpu.job
```

or directly:

```bash
python train.py \
  --dataset vggsound --num_classes 309 \
  --lr 0.0005 --lambd 0.05 \
  --num_global_views 2 --num_local_views 2 \
  --cross_modal --clean_survivor \
  --batch_size 40 --num_gpus 8 --epochs 20 \
  --num_frames 16 --frame_size 224 \
  --vit_size base --proj_dim 128 \
  --probe_lr 0.001 --probe_weight_decay 0.0 \
  --checkpoint_dir ./checkpoints
```

Useful flags:

- `--lambd`: weight of the SIGReg term in the JEPA loss `(1 - lambda) * invariance + lambda * SIGReg`.
- `--cross_modal`: alternating audio-only / video-only local views.
- `--clean_survivor`: keep the surviving modality unmasked when the other is dropped.
- `--dual_encoder`: separate audio and video encoders instead of a shared early-fusion encoder.
- `--vit_size {small,base}`: ViT-S (384d) or ViT-B (768d).
- `--attentive_probe`: online attentive probe alongside the linear probe.
- `--gradient_checkpointing`: roughly 40% activation-memory savings.

Training logs to W&B and TensorBoard; checkpoints are saved to
`--checkpoint_dir` every 2000 steps.

## Fine-tuning and probes

End-to-end fine-tuning from a pretrained checkpoint:

```bash
python finetune.py \
  --pretrained_checkpoint /path/to/pretrained.ckpt \
  --dataset vggsound \
  --lr 2e-4 --backbone_lr_scale 0.05 \
  --batch_size 160 --epochs 30 \
  --num_eval_clips 6 --attentive_probe
```

Frozen linear / attentive probe (no encoder updates):

```bash
python finetune.py \
  --pretrained_checkpoint /path/to/pretrained.ckpt \
  --dataset vggsound \
  --lr 1e-3 --backbone_lr_scale 0.0 --weight_decay 0.0 \
  --freeze_epochs 9999 --batch_size 512 --epochs 15 \
  --attentive_probe --run_test
```

Multi-label datasets (`audioset*`) automatically switch to BCE loss and report
mAP; single-label datasets use cross-entropy and top-1/top-5 accuracy.

## Cross-modal retrieval

Audio-to-video and video-to-audio retrieval on the eval split:

```bash
python retrieval.py \
  --checkpoint_path /path/to/pretrained.ckpt \
  --dataset vggsound --vit_size base \
  --use_cls --output_dir ./runs/retrieval/cls
```

Use `--use_projector` to retrieve in the projection space instead of the raw
CLS embedding, and `--draw_pairs` to save qualitative retrieval-pair figures.

## Citation

```bibtex
@inproceedings{robson2026avjepa,
  title     = {{AV-JEPA}: Extending {LeJEPA} to Audio-Visual Self-Supervised Learning},
  author    = {Robson, Benjamin and Mentu, Santeri and Zhao, Wenshuai and Solin, Arno},
  booktitle = {ICML Workshop on Machine Learning for Audio},
  year      = {2026}
}
```

## License

This code is released under the MIT license, see `LICENSE`.
