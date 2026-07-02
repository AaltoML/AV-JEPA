import os
import sys

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from fusion import AudioEmbeddings, EarlyFusionEmbeddings, VideoEmbeddings
from transformer import Encoder


class Echo(nn.Module):
    def __init__(self, a_config, v_config, t_config, gradient_checkpointing=False):
        super().__init__()
        self.t_config = t_config
        self.embedding = EarlyFusionEmbeddings(a_config, v_config, t_config)
        self.encoder = Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        self.apply(self._init_weights)

    def forward(
        self,
        video_x,
        audio_x,
        return_patches=False,
        return_attentions=False,
        video_keep_idx=None,
    ):
        x = self.embedding(video_x, audio_x, video_keep_idx=video_keep_idx)
        x, all_attentions = self.encoder(x, output_attentions=return_attentions)
        cls_token = x[:, 0]

        if return_patches:
            return cls_token, x[:, 1:]

        if return_attentions:
            return cls_token, all_attentions

        return cls_token

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Embedding)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.t_config["initializer_range"]
            )
            if (
                isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d))
                and module.bias is not None
                and not getattr(module, "_is_gate", False)
            ):
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, EarlyFusionEmbeddings):
            special_params = [
                module.video_time_embed,
                module.video_spatial_embed,
                module.audio_freq_pos_embed,
                module.audio_time_pos_embed,
                module.cls_token,
            ]

            for param in special_params:
                param.data = nn.init.trunc_normal_(
                    param.data.to(torch.float32),
                    mean=0.0,
                    std=self.t_config["initializer_range"],
                ).to(param.dtype)


class DualEcho(nn.Module):
    def __init__(self, a_config, v_config, t_config, gradient_checkpointing=False):
        super().__init__()
        self.t_config = t_config
        self.audio_embedding = AudioEmbeddings(a_config, t_config)
        self.audio_encoder = torch.compile(
            Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        )
        self.video_embedding = VideoEmbeddings(v_config, t_config)
        self.video_encoder = torch.compile(
            Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        )
        self.apply(self._init_weights)

    def forward_audio(self, audio_x, return_patches=False):
        x = self.audio_embedding(audio_x)
        x, _ = self.audio_encoder(x)
        cls_token = x[:, 0]
        if return_patches:
            return cls_token, x[:, 1:]
        return cls_token

    def forward_video(self, video_x, return_patches=False):
        x = self.video_embedding(video_x)
        x, _ = self.video_encoder(x)
        cls_token = x[:, 0]
        if return_patches:
            return cls_token, x[:, 1:]
        return cls_token

    def forward(self, video_x, audio_x, return_patches=False, return_attentions=False):
        if return_patches:
            audio_cls, audio_patches = self.forward_audio(audio_x, return_patches=True)
            video_cls, video_patches = self.forward_video(video_x, return_patches=True)
            cls_token = (audio_cls + video_cls) / 2
            patches = torch.cat([video_patches, audio_patches], dim=1)
            return cls_token, patches
        audio_cls = self.forward_audio(audio_x)
        video_cls = self.forward_video(video_x)
        return (audio_cls + video_cls) / 2

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Embedding)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.t_config["initializer_range"]
            )
            if (
                isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d))
                and module.bias is not None
                and not getattr(module, "_is_gate", False)
            ):
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, (AudioEmbeddings, VideoEmbeddings)):
            special_params = [
                p
                for name, p in module.named_parameters()
                if "pos_embed" in name
                or "time_embed" in name
                or "spatial_embed" in name
                or "freq" in name
                or "cls_token" in name
            ]
            for param in special_params:
                param.data = nn.init.trunc_normal_(
                    param.data.to(torch.float32),
                    mean=0.0,
                    std=self.t_config["initializer_range"],
                ).to(param.dtype)


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_slices=2048):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(-5, 5, knots, dtype=torch.float32)
        window = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", window)

    def forward(self, proj, global_step):
        dev = proj.device
        g = torch.Generator(device=dev)
        g.manual_seed(global_step)

        A = torch.randn(proj.size(-1), self.num_slices, generator=g, device=dev)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj.float() @ A).unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)

        if dist.is_initialized():
            dist.all_reduce(cos_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(sin_mean, op=dist.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        n_global = proj.size(0) * world_size

        statistic = torch.trapz(err * self.phi, self.t, dim=-1) * n_global
        return statistic.mean()


class VISReg(torch.nn.Module):
    """Variance-Invariance-Sketching Regularization (arXiv:2606.02572).

    Drop-in replacement for ``SIGReg``: takes one view's projected batch
    ``proj`` of shape ``[N, D]`` and returns the scalar regularizer
    ``w_scale*L_scale + w_shape*L_shape + w_center*L_center``.

    - L_scale  : (1 - std).pow(2) per dim -> pushes each dim to unit variance.
                 Unlike SIGReg's characteristic-function test, the gradient stays
                 ~constant (-2) as std -> 0, so it keeps correcting under collapse.
    - L_shape  : sliced-Wasserstein "sketch" -- project the std-normalized,
                 centered embeddings onto K random unit directions, sort each 1-D
                 marginal, and match it to standard-Gaussian quantiles (Cramer-Wold).
    - L_center : ||mean||^2, a small anti-drift term.

    Computed locally per-GPU (paper Algorithm 1); DDP averages the gradients.
    The shape projections are seeded by (global_step, rank) so every rank draws
    INDEPENDENT directions -> K * world_size effective slices, no all_gather.
    """

    def __init__(self, num_slices=2048, w_scale=1.0, w_shape=1.0, w_center=1.0, eps=1e-6):
        super().__init__()
        self.num_slices = num_slices
        self.w_scale = w_scale
        self.w_shape = w_shape
        self.w_center = w_center
        self.eps = eps
        self.last_parts = {}

    def forward(self, proj, global_step):
        z = proj.float()
        N, D = z.shape
        dev = z.device

        mu = z.mean(dim=0)
        l_center = mu.square().mean()
        z_cent = z - mu

        std = z_cent.std(dim=0, unbiased=False)
        l_scale = (1.0 - std).square().mean()

        z_norm = z_cent / (std.detach() + self.eps)
        rank = dist.get_rank() if dist.is_initialized() else 0
        g = torch.Generator(device=dev)
        g.manual_seed(int(global_step) * 100003 + rank)
        A = torch.randn(D, self.num_slices, generator=g, device=dev)
        A = A / A.norm(p=2, dim=0)
        p_sorted = torch.sort(z_norm @ A, dim=0).values
        u = torch.arange(1, N + 1, device=dev, dtype=z.dtype) / (N + 1)
        q = torch.special.ndtri(u).unsqueeze(1)
        l_shape = (p_sorted - q).square().mean()

        self.last_parts = {
            "scale": l_scale.detach(),
            "shape": l_shape.detach(),
            "center": l_center.detach(),
        }
        return self.w_scale * l_scale + self.w_shape * l_shape + self.w_center * l_center


class Projector(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, proj_dim=128, pred_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentiveProbe(nn.Module):
    def __init__(self, hidden_size, num_heads, num_classes):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, patches):
        query = self.query.expand(patches.size(0), -1, -1)
        out, _ = self.cross_attn(query, patches, patches)
        return self.head(self.norm(out.squeeze(1)))


def modality_tokens(patches, num_video_tokens, modality):
    """Select one modality's tokens from fused patch tokens (CLS stripped).

    The attentive probe has no key masking, so a zeroed modality still
    contributes content-free keys (conv bias + positional + type embeddings)
    that soak up attention; unimodal eval must slice them out. Token layout
    is [video, audio]. num_video_tokens=None (no early-fusion layout, e.g.
    DualEcho) returns patches unchanged.
    """
    if num_video_tokens is None:
        return patches
    if modality == "video":
        return patches[:, :num_video_tokens]
    return patches[:, num_video_tokens:]


class EchoTrainer(L.LightningModule):
    def __init__(
        self,
        a_config,
        v_config,
        t_config,
        lr=0.0003,
        weight_decay=5e-2,
        lambd=0.02,
        num_views=2,
        batch_size=32,
        epochs=5,
        proj_dim=128,
        num_classes: int = 309,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 0.0,
        cross_modal: bool = False,
        per_modal_loss: bool = False,
        video_token_crop: bool = False,
        gradient_checkpointing: bool = False,
        attentive_probe: bool = False,
        dual_encoder: bool = False,
        mean_pool: bool = False,
        predictive: bool = False,
        pred_dim: int = 64,
        multi_label: bool = False,
        total_samples: int = 183_730,
        pos_weight: torch.Tensor = None,
        mm_sigreg: bool = False,
        mm_sigreg_num_slices: int = 2048,
        mm_sigreg_w_marg: float = 1.0,
        mm_sigreg_w_joint: float = 0.5,
        mm_sigreg_w_ind: float = 0.25,
        sep_loss: bool = False,
        sep_weight: float = 0.1,
        sep_start_step: int = 5000,
        sep_ramp_steps: int = 2000,
        visreg: bool = False,
        visreg_num_slices: int = 2048,
        visreg_w_scale: float = 1.0,
        visreg_w_shape: float = 1.0,
        visreg_w_center: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.cross_modal = cross_modal
        self.per_modal_loss = per_modal_loss
        self.attentive_probe = attentive_probe
        self.mean_pool = mean_pool
        self.dual_encoder = dual_encoder
        self.predictive = predictive
        self.multi_label = multi_label
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        if dual_encoder:
            self.encoder = DualEcho(
                a_config,
                v_config,
                t_config,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.num_video_tokens = None
        else:
            self.encoder = torch.compile(
                Echo(
                    a_config,
                    v_config,
                    t_config,
                    gradient_checkpointing=gradient_checkpointing,
                ),
                dynamic=False if video_token_crop else None,
            )
            self.num_video_tokens = (
                v_config["num_frames"] // v_config["tubelet_size"]
            ) * (v_config["image_size"] // v_config["patch_size"]) ** 2
        self.projector = torch.compile(
            Projector(
                in_dim=t_config["hidden_size"],
                hidden_dim=2048,
                out_dim=proj_dim,
            )
        )
        if predictive:
            self.predictor_head = torch.compile(Predictor(proj_dim, pred_dim))
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambd = lambd
        self.num_views = num_views

        self.total_samples = total_samples
        self.epochs = epochs

        self.sigreg = SIGReg()
        self.sep_enabled = bool(sep_loss)
        self.sep_weight = sep_weight
        self.sep_start_step = sep_start_step
        self.sep_ramp_steps = sep_ramp_steps
        self.mm_sigreg_enabled = bool(mm_sigreg)
        if self.mm_sigreg_enabled:
            from multimodal_sigreg import MultimodalSIGReg

            self.mm_sigreg = MultimodalSIGReg(
                num_slices=mm_sigreg_num_slices,
                w_marg=mm_sigreg_w_marg,
                w_joint=mm_sigreg_w_joint,
                w_ind=mm_sigreg_w_ind,
            )
        else:
            self.mm_sigreg = None

        self.visreg_enabled = bool(visreg)
        if self.visreg_enabled:
            assert not self.mm_sigreg_enabled, (
                "--visreg and --mm_sigreg are mutually exclusive"
            )
            self.visreg = VISReg(
                num_slices=visreg_num_slices,
                w_scale=visreg_w_scale,
                w_shape=visreg_w_shape,
                w_center=visreg_w_center,
            )
        else:
            self.visreg = None

        hidden_size = t_config["hidden_size"]
        self.probe_norm = torch.compile(nn.LayerNorm(hidden_size))
        self.probe_head = torch.compile(nn.Linear(hidden_size, num_classes))
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay

        if multi_label:
            self.probe_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            self.probe_train_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.probe_test_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.probe_test_audio_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.probe_test_video_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
        else:
            self.probe_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
            self.probe_train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.probe_train_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.probe_test_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.probe_test_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.probe_test_audio_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.probe_test_audio_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.probe_test_video_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.probe_test_video_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )

        if attentive_probe:
            self.att_probe = torch.compile(
                AttentiveProbe(
                    hidden_size, t_config["num_attention_heads"], num_classes
                )
            )
            if multi_label:
                self.att_probe_criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weight
                )
                self.att_probe_train_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_audio_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_video_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
            else:
                self.att_probe_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                self.att_probe_train_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_train_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_audio_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_audio_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_video_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_video_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )

        if cross_modal:
            if multi_label:
                self.probe_audio_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.probe_video_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
            else:
                self.probe_audio_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.probe_video_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.probe_audio_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.probe_video_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )

    def forward(self, video, audio):
        return self.encoder(video, audio)

    def _reg_term(self, emb, step):
        """Per-view embedding-space regularizer: VISReg when enabled, else SIGReg.

        Both take a ``[N, proj_dim]`` projected batch and return a scalar, so this
        is a drop-in swap at every SIGReg call site.
        """
        if self.visreg_enabled:
            return self.visreg(emb, step)
        return self.sigreg(emb, step)

    def _per_modal_sigreg(self, audio_embs, video_embs):
        """Compute the SIGReg term over per-modality embedding bags.

        Returns (sigreg_loss, parts_or_None). When MM-SIGReg is enabled we
        delegate to it (joint + independence terms in addition to marginal
        isotropy); otherwise we fall back to the LeJEPA-style per-modality
        statistic averaged across the two modalities.
        """
        if self.mm_sigreg_enabled:
            return self.mm_sigreg(audio_embs, video_embs, self.global_step)
        sigreg_audio = torch.stack(
            [self._reg_term(emb, self.global_step) for emb in audio_embs]
        ).mean()
        sigreg_video = torch.stack(
            [self._reg_term(emb, self.global_step) for emb in video_embs]
        ).mean()
        return (sigreg_audio + sigreg_video) / 2, None

    def _cross_modal_separation(
        self,
        centers,
        cls_global,
        global_video_flat,
        global_spectrogram_flat,
        B,
        G,
        eps=1e-6,
    ):
        """Cross-modal-disagreement repulsion on per-sample fused embeddings.

        Pushes apart pairs the FUSED encoder thinks are alike (high fused-CLS
        cosine) but at least one single modality says are different (low
        audio-only OR video-only CLS cosine). The complementary modality is a
        label-free 'are these really the same class?' gate: pairs both
        modalities agree on (genuine same-class) are protected, spuriously
        merged super-cluster pairs are separated. This sits in the niche SIGReg
        is blind to -- SIGReg fixes the global (isotropic-Gaussian) envelope but
        not how classes are partitioned inside it -- which is exactly where the
        plain-uniformity KoLeo ablation (job 18832216) had no headroom.

        The gate is built in CLS space under no_grad: the encoder is LayerNorm
        (no running stats), so the two extra modality-zeroed forwards have no
        side effects, unlike routing them through the BatchNorm projector. The
        repulsion gradient flows ONLY through the projected `centers`, keeping
        the SSL geometry in projector space alongside invariance/SIGReg. Local
        batch only (no all-reduce), matching the DINOv2/KoLeo convention.

        Returns (sep_loss, gate_fraction). All [B, B] matrices are indexed by
        sample, consistent with `centers` (= projected global-view mean).
        """
        with torch.no_grad():
            fused = F.normalize(cls_global.view(B, G, -1).mean(dim=1).float(), dim=1)
            audio_only = self.encoder(
                torch.zeros_like(global_video_flat), global_spectrogram_flat
            )
            video_only = self.encoder(
                global_video_flat, torch.zeros_like(global_spectrogram_flat)
            )
            u_audio = F.normalize(
                audio_only.view(B, G, -1).mean(dim=1).float(), dim=1
            )
            u_video = F.normalize(
                video_only.view(B, G, -1).mean(dim=1).float(), dim=1
            )
            s_fused = fused @ fused.t()
            s_audio = u_audio @ u_audio.t()
            s_video = u_video @ u_video.t()
            gate = s_fused.clamp_min(0.0) * (
                1.0 - torch.minimum(s_audio, s_video)
            ).clamp_min(0.0)
            gate.fill_diagonal_(0.0)

        u_fused = F.normalize(centers.float(), dim=1)
        sim = u_fused @ u_fused.t()
        denom = gate.sum().clamp_min(eps)
        sep = (gate * sim).sum() / denom
        gate_frac = (gate > 0).float().mean()
        return sep, gate_frac

    def training_step(self, batch, batch_idx):
        if self.predictive:
            return self._predictive_step(batch, batch_idx)

        global_video = batch["global_video"]
        global_spectrogram = batch["global_spectrogram"]

        local_video = batch["local_video"]
        local_spectrogram = batch["local_spectrogram"]

        B, G = global_video.shape[:2]
        K = local_video.shape[1]

        global_video_flat = global_video.view(B * G, *global_video.shape[2:])
        global_spectrogram_flat = global_spectrogram.view(
            B * G, *global_spectrogram.shape[2:]
        )
        if self.dual_encoder:
            if self.attentive_probe or self.mean_pool:
                audio_cls_g, audio_patches_g = self.encoder.forward_audio(
                    global_spectrogram_flat, return_patches=True
                )
                video_cls_g, video_patches_g = self.encoder.forward_video(
                    global_video_flat, return_patches=True
                )
                patches_global = torch.cat([video_patches_g, audio_patches_g], dim=1)
            else:
                audio_cls_g = self.encoder.forward_audio(global_spectrogram_flat)
                video_cls_g = self.encoder.forward_video(global_video_flat)
            cls_global = (audio_cls_g + video_cls_g) / 2
        else:
            if self.attentive_probe or self.mean_pool:
                cls_global, patches_global = self.encoder(
                    global_video_flat, global_spectrogram_flat, return_patches=True
                )
            else:
                cls_global = self.encoder(
                    global_video_flat, global_spectrogram_flat
                )
        z_global_all = cls_global

        if self.mean_pool:
            probe_features = (
                patches_global.view(B, G, *patches_global.shape[1:])
                .mean(dim=1)
                .mean(dim=1)
                .detach()
            )
        else:
            probe_features = cls_global.view(B, G, -1).mean(dim=1).detach()
        probe_logits = self.probe_head(self.probe_norm(probe_features))
        labels = batch["label"]
        probe_loss = self.probe_criterion(probe_logits, labels)
        if self.multi_label:
            self.probe_train_map(probe_logits, labels.long())
        else:
            self.probe_train_acc(probe_logits.argmax(dim=1), labels)
            self.probe_train_acc5(probe_logits, labels)

        if self.attentive_probe:
            patches_for_att = (
                patches_global.view(B, G, *patches_global.shape[1:])
                .mean(dim=1)
                .detach()
            )
            att_logits = self.att_probe(patches_for_att)
            att_probe_loss = self.att_probe_criterion(att_logits, labels)
            if self.multi_label:
                self.att_probe_train_map(att_logits, labels.long())
            else:
                self.att_probe_train_acc(att_logits.argmax(dim=1), labels)
                self.att_probe_train_acc5(att_logits, labels)

        if self.dual_encoder and self.cross_modal:
            Ka = K // 2
            Kv = K - Ka
            audio_spec = local_spectrogram[:, 0::2].reshape(
                B * Ka, *local_spectrogram.shape[2:]
            )
            video_vid = local_video[:, 1::2].reshape(B * Kv, *local_video.shape[2:])
            z_audio_local = self.encoder.forward_audio(audio_spec)
            z_video_local = self.encoder.forward_video(video_vid)

            audio_cls = z_audio_local.view(B, Ka, -1).mean(dim=1).detach()
            video_cls = z_video_local.view(B, Kv, -1).mean(dim=1).detach()
            audio_logits = self.probe_head(self.probe_norm(audio_cls))
            video_logits = self.probe_head(self.probe_norm(video_cls))
            audio_probe_loss = self.probe_criterion(audio_logits, labels)
            video_probe_loss = self.probe_criterion(video_logits, labels)
            unimodal_probe_loss = 0.5 * (audio_probe_loss + video_probe_loss)
            self.log("train/probe_audio_loss", audio_probe_loss, sync_dist=True)
            self.log("train/probe_video_loss", video_probe_loss, sync_dist=True)
            if self.multi_label:
                self.probe_audio_map(audio_logits, labels.long())
                self.probe_video_map(video_logits, labels.long())
                self.log("train/probe_audio_map", self.probe_audio_map, sync_dist=True)
                self.log("train/probe_video_map", self.probe_video_map, sync_dist=True)
            else:
                self.probe_audio_acc(audio_logits.argmax(dim=1), labels)
                self.probe_video_acc(video_logits.argmax(dim=1), labels)
                self.probe_audio_acc5(audio_logits, labels)
                self.probe_video_acc5(video_logits, labels)
                self.log("train/probe_audio_acc", self.probe_audio_acc, sync_dist=True)
                self.log("train/probe_video_acc", self.probe_video_acc, sync_dist=True)
                self.log(
                    "train/probe_audio_acc_top5", self.probe_audio_acc5, sync_dist=True
                )
                self.log(
                    "train/probe_video_acc_top5", self.probe_video_acc5, sync_dist=True
                )

            z_audio_global = self.projector(audio_cls_g)
            z_video_global = self.projector(video_cls_g)
            z_audio_global = z_audio_global.view(B, G, -1).permute(
                1, 0, 2
            )
            z_video_global = z_video_global.view(B, G, -1).permute(
                1, 0, 2
            )

            z_audio_local_proj = self.projector(z_audio_local)
            z_video_local_proj = self.projector(z_video_local)
            z_audio_local_proj = z_audio_local_proj.view(B, Ka, -1).permute(
                1, 0, 2
            )
            z_video_local_proj = z_video_local_proj.view(B, Kv, -1).permute(
                1, 0, 2
            )

            audio_embs = torch.cat(
                [z_audio_global, z_audio_local_proj], dim=0
            )
            video_embs = torch.cat(
                [z_video_global, z_video_local_proj], dim=0
            )

            audio_center = z_audio_global.mean(dim=0)
            video_center = z_video_global.mean(dim=0)
            inv_loss = (
                (video_center - audio_embs).square().mean()
                + (audio_center - video_embs).square().mean()
            ) / 2

            sigreg_loss, sigreg_parts = self._per_modal_sigreg(audio_embs, video_embs)

            embed_std = (
                audio_embs.std(dim=1).mean() + video_embs.std(dim=1).mean()
            ) / 2
        else:
            if not (self.per_modal_loss and self.cross_modal):
                z_global_all = self.projector(z_global_all)
                z_global_all = z_global_all.view(B, G, -1).permute(
                    1, 0, 2
                )

            local_video = local_video.view(B * K, *local_video.shape[2:])
            local_spectrogram = local_spectrogram.view(
                B * K, *local_spectrogram.shape[2:]
            )
            local_keep_idx = batch.get("local_video_keep_idx")
            if local_keep_idx is not None:
                local_keep_idx = local_keep_idx.view(B * K, -1)
            z_local_all = self.encoder(
                local_video, local_spectrogram, video_keep_idx=local_keep_idx
            )

            if self.cross_modal:
                local_cls = z_local_all.view(B, K, -1).detach()
                audio_cls = local_cls[:, 0::2, :].mean(dim=1)
                video_cls = local_cls[:, 1::2, :].mean(dim=1)
                audio_logits = self.probe_head(self.probe_norm(audio_cls))
                video_logits = self.probe_head(self.probe_norm(video_cls))
                audio_probe_loss = self.probe_criterion(audio_logits, labels)
                video_probe_loss = self.probe_criterion(video_logits, labels)
                unimodal_probe_loss = 0.5 * (audio_probe_loss + video_probe_loss)
                self.log("train/probe_audio_loss", audio_probe_loss, sync_dist=True)
                self.log("train/probe_video_loss", video_probe_loss, sync_dist=True)
                if self.multi_label:
                    self.probe_audio_map(audio_logits, labels.long())
                    self.probe_video_map(video_logits, labels.long())
                    self.log(
                        "train/probe_audio_map", self.probe_audio_map, sync_dist=True
                    )
                    self.log(
                        "train/probe_video_map", self.probe_video_map, sync_dist=True
                    )
                else:
                    self.probe_audio_acc(audio_logits.argmax(dim=1), labels)
                    self.probe_video_acc(video_logits.argmax(dim=1), labels)
                    self.probe_audio_acc5(audio_logits, labels)
                    self.probe_video_acc5(video_logits, labels)
                    self.log(
                        "train/probe_audio_acc", self.probe_audio_acc, sync_dist=True
                    )
                    self.log(
                        "train/probe_video_acc", self.probe_video_acc, sync_dist=True
                    )
                    self.log(
                        "train/probe_audio_acc_top5",
                        self.probe_audio_acc5,
                        sync_dist=True,
                    )
                    self.log(
                        "train/probe_video_acc_top5",
                        self.probe_video_acc5,
                        sync_dist=True,
                    )
            else:
                unimodal_probe_loss = None

            z_local_all = self.projector(z_local_all)
            z_local_all = z_local_all.view(B, K, -1).permute(
                1, 0, 2
            )

            if self.per_modal_loss and self.cross_modal:
                audio_only_g = self.encoder(
                    torch.zeros_like(global_video_flat), global_spectrogram_flat
                )
                video_only_g = self.encoder(
                    global_video_flat, torch.zeros_like(global_spectrogram_flat)
                )
                z_audio_global = (
                    self.projector(audio_only_g).view(B, G, -1).permute(1, 0, 2)
                )
                z_video_global = (
                    self.projector(video_only_g).view(B, G, -1).permute(1, 0, 2)
                )

                z_audio_local = z_local_all[0::2]
                z_video_local = z_local_all[1::2]
                audio_embs = torch.cat([z_audio_global, z_audio_local], dim=0)
                video_embs = torch.cat([z_video_global, z_video_local], dim=0)
                audio_center = z_audio_global.mean(dim=0)
                video_center = z_video_global.mean(dim=0)

                inv_loss = (
                    (video_center - audio_embs).square().mean()
                    + (audio_center - video_embs).square().mean()
                ) / 2
                sigreg_loss, sigreg_parts = self._per_modal_sigreg(
                    audio_embs, video_embs
                )
                embed_std = (
                    audio_embs.std(dim=1).mean() + video_embs.std(dim=1).mean()
                ) / 2
            else:
                centers = z_global_all.mean(dim=0)
                a_emb = torch.cat([z_global_all, z_local_all], dim=0)
                inv_loss = (centers - a_emb).square().mean()
                sigreg_loss = torch.stack(
                    [self._reg_term(emb, self.global_step) for emb in a_emb]
                ).mean()
                sigreg_parts = None
                embed_std = a_emb.std(dim=1).mean()

        loss = (1 - self.lambd) * inv_loss + self.lambd * sigreg_loss
        if (
            self.sep_enabled
            and not self.dual_encoder
            and not (self.per_modal_loss and self.cross_modal)
            and self.global_step >= self.sep_start_step
        ):
            w_sep = self.sep_weight * min(
                1.0,
                (self.global_step - self.sep_start_step)
                / max(1, self.sep_ramp_steps),
            )
            sep_loss_val, sep_gate_frac = self._cross_modal_separation(
                centers, cls_global, global_video_flat, global_spectrogram_flat, B, G
            )
            loss = loss + w_sep * sep_loss_val
            self.log("train/sep_loss", sep_loss_val, sync_dist=True)
            self.log("train/sep_loss_weighted", w_sep * sep_loss_val, sync_dist=True)
            self.log("train/sep_weight", w_sep, sync_dist=True)
            self.log("train/sep_gate_frac", sep_gate_frac, sync_dist=True)
        self.log("train/embed_std", embed_std, sync_dist=True)
        self.log("train/lejepa_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/inv_loss", inv_loss, sync_dist=True)
        self.log("train/sigreg_loss", sigreg_loss, sync_dist=True)
        self.log("train/inv_loss_weighted", (1 - self.lambd) * inv_loss, sync_dist=True)
        self.log("train/sigreg_loss_weighted", self.lambd * sigreg_loss, sync_dist=True)
        if sigreg_parts is not None:
            for _k, _v in sigreg_parts.items():
                self.log(f"train/sigreg_{_k}", _v, sync_dist=True)
        if self.visreg_enabled and self.visreg.last_parts:
            for _k, _v in self.visreg.last_parts.items():
                self.log(f"train/visreg_{_k}", _v, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        self.log("train/probe_loss", probe_loss, sync_dist=True)
        if self.multi_label:
            self.log("train/probe_map", self.probe_train_map, sync_dist=True)
        else:
            self.log("train/probe_acc", self.probe_train_acc, sync_dist=True)
            self.log("train/probe_acc_top5", self.probe_train_acc5, sync_dist=True)

        total_loss = loss + probe_loss
        if self.cross_modal and unimodal_probe_loss is not None:
            total_loss = total_loss + unimodal_probe_loss
            self.log("train/probe_unimodal_loss", unimodal_probe_loss, sync_dist=True)
        if self.attentive_probe:
            total_loss = total_loss + att_probe_loss
            self.log("train/att_probe_loss", att_probe_loss, sync_dist=True)
            if self.multi_label:
                self.log(
                    "train/att_probe_map", self.att_probe_train_map, sync_dist=True
                )
            else:
                self.log(
                    "train/att_probe_acc", self.att_probe_train_acc, sync_dist=True
                )
                self.log(
                    "train/att_probe_acc_top5",
                    self.att_probe_train_acc5,
                    sync_dist=True,
                )

        self.log("train/joint_loss", total_loss, sync_dist=True)
        return total_loss

    def _predictive_step(self, batch, batch_idx):
        local_video = batch["local_video"]
        local_spectrogram = batch["local_spectrogram"]
        B = local_video.shape[0]
        labels = batch["label"]

        if self.attentive_probe or self.mean_pool:
            z_a, patches_a = self.encoder(
                local_video[:, 0], local_spectrogram[:, 0], return_patches=True
            )
            z_v, patches_v = self.encoder(
                local_video[:, 1], local_spectrogram[:, 1], return_patches=True
            )
        else:
            z_a = self.encoder(local_video[:, 0], local_spectrogram[:, 0])
            z_v = self.encoder(local_video[:, 1], local_spectrogram[:, 1])

        if self.mean_pool:
            probe_features = ((patches_a + patches_v) / 2).mean(dim=1).detach()
        else:
            probe_features = ((z_a + z_v) / 2).detach()
        probe_logits = self.probe_head(self.probe_norm(probe_features))
        probe_loss = self.probe_criterion(probe_logits, labels)
        if self.multi_label:
            self.probe_train_map(probe_logits, labels.long())
        else:
            self.probe_train_acc(probe_logits.argmax(dim=1), labels)
            self.probe_train_acc5(probe_logits, labels)

        audio_logits = self.probe_head(self.probe_norm(z_a.detach()))
        video_logits = self.probe_head(self.probe_norm(z_v.detach()))
        if self.multi_label:
            self.probe_audio_map(audio_logits, labels.long())
            self.probe_video_map(video_logits, labels.long())
            self.log("train/probe_audio_map", self.probe_audio_map, sync_dist=True)
            self.log("train/probe_video_map", self.probe_video_map, sync_dist=True)
        else:
            self.probe_audio_acc(audio_logits.argmax(dim=1), labels)
            self.probe_video_acc(video_logits.argmax(dim=1), labels)
            self.probe_audio_acc5(audio_logits, labels)
            self.probe_video_acc5(video_logits, labels)
            self.log("train/probe_audio_acc", self.probe_audio_acc, sync_dist=True)
            self.log("train/probe_video_acc", self.probe_video_acc, sync_dist=True)
            self.log(
                "train/probe_audio_acc_top5", self.probe_audio_acc5, sync_dist=True
            )
            self.log(
                "train/probe_video_acc_top5", self.probe_video_acc5, sync_dist=True
            )

        if self.attentive_probe:
            att_patches = ((patches_a + patches_v) / 2).detach()
            att_logits = self.att_probe(att_patches)
            att_probe_loss = self.att_probe_criterion(att_logits, labels)
            if self.multi_label:
                self.att_probe_train_map(att_logits, labels.long())
            else:
                self.att_probe_train_acc(att_logits.argmax(dim=1), labels)
                self.att_probe_train_acc5(att_logits, labels)

        h_a = self.projector(z_a)
        h_v = self.projector(z_v)

        p_a2v = self.predictor_head(h_a)
        p_v2a = self.predictor_head(h_v)
        inv_loss = (
            F.mse_loss(p_a2v, h_v.detach()) + F.mse_loss(p_v2a, h_a.detach())
        ) / 2

        sigreg_loss, sigreg_parts = self._per_modal_sigreg(
            h_a.unsqueeze(0), h_v.unsqueeze(0)
        )

        loss = (1 - self.lambd) * inv_loss + self.lambd * sigreg_loss

        embed_std = torch.stack([h_a.std(dim=0), h_v.std(dim=0)]).mean()
        self.log("train/embed_std", embed_std, sync_dist=True)
        self.log("train/lejepa_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/inv_loss", inv_loss, sync_dist=True)
        self.log("train/sigreg_loss", sigreg_loss, sync_dist=True)
        self.log("train/inv_loss_weighted", (1 - self.lambd) * inv_loss, sync_dist=True)
        self.log("train/sigreg_loss_weighted", self.lambd * sigreg_loss, sync_dist=True)
        if sigreg_parts is not None:
            for _k, _v in sigreg_parts.items():
                self.log(f"train/sigreg_{_k}", _v, sync_dist=True)
        if self.visreg_enabled and self.visreg.last_parts:
            for _k, _v in self.visreg.last_parts.items():
                self.log(f"train/visreg_{_k}", _v, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        self.log("train/probe_loss", probe_loss, sync_dist=True)
        if self.multi_label:
            self.log("train/probe_map", self.probe_train_map, sync_dist=True)
        else:
            self.log("train/probe_acc", self.probe_train_acc, sync_dist=True)
            self.log("train/probe_acc_top5", self.probe_train_acc5, sync_dist=True)

        total_loss = loss + probe_loss
        if self.attentive_probe:
            total_loss = total_loss + att_probe_loss
            self.log("train/att_probe_loss", att_probe_loss, sync_dist=True)
            if self.multi_label:
                self.log(
                    "train/att_probe_map", self.att_probe_train_map, sync_dist=True
                )
            else:
                self.log(
                    "train/att_probe_acc", self.att_probe_train_acc, sync_dist=True
                )
                self.log(
                    "train/att_probe_acc_top5",
                    self.att_probe_train_acc5,
                    sync_dist=True,
                )

        self.log("train/joint_loss", total_loss, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        linear_probe_params = list(self.probe_head.parameters()) + list(
            self.probe_norm.parameters()
        )
        att_probe_params = (
            list(self.att_probe.parameters()) if self.attentive_probe else []
        )
        all_probe_params = linear_probe_params + att_probe_params
        probe_ids = {id(p) for p in all_probe_params}
        backbone_params = [p for p in self.parameters() if id(p) not in probe_ids]

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": linear_probe_params,
                "lr": self.probe_lr,
                "weight_decay": self.probe_weight_decay,
            },
        ]
        if self.attentive_probe:
            param_groups.append(
                {
                    "params": att_probe_params,
                    "lr": self.probe_lr,
                    "weight_decay": self.probe_weight_decay,
                }
            )

        optimizer = optim.AdamW(param_groups)

        accumulation_steps = self.trainer.accumulate_grad_batches or 1
        effective_batch_size = (
            self.hparams.batch_size * self.trainer.world_size * accumulation_steps
        )

        steps_per_epoch = self.total_samples // effective_batch_size
        total_steps = steps_per_epoch * self.trainer.max_epochs

        warmup_steps = int(total_steps * 0.15)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )

        t_max = max(1, total_steps - warmup_steps)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-6
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def test_step(self, batch, batch_idx):
        video = batch["video"]
        audio = batch["spectrogram"]
        targets = batch["label"]
        B, N = video.shape[:2]
        video = video.view(B * N, *video.shape[2:])
        audio = audio.view(B * N, *audio.shape[2:])
        with torch.no_grad():
            if self.attentive_probe or self.mean_pool:
                cls_tokens, patch_tokens = self.encoder(
                    video, audio, return_patches=True
                )
            else:
                cls_tokens = self.encoder(video, audio)
            if self.mean_pool:
                probe_features = patch_tokens.mean(dim=1)
            else:
                probe_features = cls_tokens
            logits = self.probe_head(
                self.probe_norm(probe_features)
            )
            logits = logits.view(B, N, -1).mean(dim=1)
        loss = self.probe_criterion(logits, targets)
        self.log("probe/test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.multi_label:
            self.probe_test_map(logits, targets.long())
            self.log(
                "probe/test/map",
                self.probe_test_map,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            self.probe_test_acc(logits.argmax(dim=1), targets)
            self.probe_test_acc5(logits, targets)
            self.log(
                "probe/test/acc",
                self.probe_test_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "probe/test/acc_top5",
                self.probe_test_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.attentive_probe:
            with torch.no_grad():
                att_logits = self.att_probe(patch_tokens)
                att_logits = att_logits.view(B, N, -1).mean(dim=1)
            att_loss = self.att_probe_criterion(att_logits, targets)
            self.log(
                "att_probe/test/loss",
                att_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if self.multi_label:
                self.att_probe_test_map(att_logits, targets.long())
                self.log(
                    "att_probe/test/map",
                    self.att_probe_test_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.att_probe_test_acc(att_logits.argmax(dim=1), targets)
                self.att_probe_test_acc5(att_logits, targets)
                self.log(
                    "att_probe/test/acc",
                    self.att_probe_test_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "att_probe/test/acc_top5",
                    self.att_probe_test_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        with torch.no_grad():
            video_zero = torch.zeros_like(video)
            audio_zero = torch.zeros_like(audio)

            if self.attentive_probe or self.mean_pool:
                audio_cls, audio_patches = self.encoder(
                    video_zero, audio, return_patches=True
                )
                video_cls, video_patches = self.encoder(
                    video, audio_zero, return_patches=True
                )
            else:
                audio_cls = self.encoder(video_zero, audio)
                video_cls = self.encoder(video, audio_zero)

            if self.mean_pool:
                audio_features = audio_patches.mean(dim=1)
                video_features = video_patches.mean(dim=1)
            else:
                audio_features = audio_cls
                video_features = video_cls

            audio_logits = self.probe_head(self.probe_norm(audio_features))
            audio_logits = audio_logits.view(B, N, -1).mean(dim=1)
            video_logits = self.probe_head(self.probe_norm(video_features))
            video_logits = video_logits.view(B, N, -1).mean(dim=1)

        if self.multi_label:
            self.probe_test_audio_map(audio_logits, targets.long())
            self.probe_test_video_map(video_logits, targets.long())
            self.log(
                "probe/test/audio_map",
                self.probe_test_audio_map,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "probe/test/video_map",
                self.probe_test_video_map,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            self.probe_test_audio_acc(audio_logits.argmax(dim=1), targets)
            self.probe_test_audio_acc5(audio_logits, targets)
            self.probe_test_video_acc(video_logits.argmax(dim=1), targets)
            self.probe_test_video_acc5(video_logits, targets)
            self.log(
                "probe/test/audio_acc",
                self.probe_test_audio_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "probe/test/audio_acc_top5",
                self.probe_test_audio_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "probe/test/video_acc",
                self.probe_test_video_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "probe/test/video_acc_top5",
                self.probe_test_video_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.attentive_probe:
            att_audio_logits = (
                self.att_probe(
                    modality_tokens(audio_patches, self.num_video_tokens, "audio")
                )
                .view(B, N, -1)
                .mean(dim=1)
            )
            att_video_logits = (
                self.att_probe(
                    modality_tokens(video_patches, self.num_video_tokens, "video")
                )
                .view(B, N, -1)
                .mean(dim=1)
            )
            if self.multi_label:
                self.att_probe_test_audio_map(att_audio_logits, targets.long())
                self.att_probe_test_video_map(att_video_logits, targets.long())
                self.log(
                    "att_probe/test/audio_map",
                    self.att_probe_test_audio_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "att_probe/test/video_map",
                    self.att_probe_test_video_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.att_probe_test_audio_acc(att_audio_logits.argmax(dim=1), targets)
                self.att_probe_test_audio_acc5(att_audio_logits, targets)
                self.att_probe_test_video_acc(att_video_logits.argmax(dim=1), targets)
                self.att_probe_test_video_acc5(att_video_logits, targets)
                self.log(
                    "att_probe/test/audio_acc",
                    self.att_probe_test_audio_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "att_probe/test/audio_acc_top5",
                    self.att_probe_test_audio_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "att_probe/test/video_acc",
                    self.att_probe_test_video_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "att_probe/test/video_acc_top5",
                    self.att_probe_test_video_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss


class EchoFineTuner(L.LightningModule):
    """Supervised finetuning: loads a pretrained encoder and trains end-to-end.

    multi_label=False (VGGSound): cross-entropy + top-1/top-5 accuracy.
    multi_label=True (AudioSet): BCEWithLogits (with optional pos_weight) + mAP.
    Mixup mixes audio and video with a shared lambda from Beta(alpha, alpha);
    multi-label mixes the multi-hot targets, single-label combines the CE
    losses of both endpoint labels.
    """

    def __init__(
        self,
        encoder,
        hidden_size,
        num_classes=309,
        lr=1e-4,
        backbone_lr_scale=0.1,
        weight_decay=5e-2,
        warmup_fraction=0.05,
        label_smoothing=0.1,
        batch_size=32,
        total_samples=183_730,
        epochs=10,
        freeze_epochs=0,
        attentive_probe=False,
        mean_pool=False,
        num_attention_heads=12,
        multi_label=False,
        pos_weight=None,
        mixup_alpha=0.0,
        modality_drop_prob=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "pos_weight"])
        self.encoder = encoder
        emb = getattr(encoder, "embedding", None)
        if emb is not None and hasattr(emb, "num_temporal_patches"):
            self.num_video_tokens = emb.num_temporal_patches * emb.num_spatial_patches
        else:
            self.num_video_tokens = None
        self.attentive_probe = attentive_probe
        self.mean_pool = mean_pool
        self.multi_label = multi_label
        self.mixup_alpha = mixup_alpha
        self.modality_drop_prob = modality_drop_prob

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        if multi_label:
            if pos_weight is not None:
                self.register_buffer("pos_weight", pos_weight.clone().detach())
            else:
                self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.lr = lr
        self.backbone_lr_scale = backbone_lr_scale
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.total_samples = total_samples
        self.epochs = epochs
        self.freeze_epochs = freeze_epochs

        if multi_label:
            self.train_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.train_audio_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.train_video_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.test_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.test_audio_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
            self.test_video_map = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
        else:
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.train_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.train_audio_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.train_audio_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.train_video_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.train_video_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.test_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.test_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.test_audio_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.test_audio_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
            self.test_video_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.test_video_acc5 = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )

        if attentive_probe:
            self.att_probe = AttentiveProbe(
                hidden_size, num_attention_heads, num_classes
            )
            if multi_label:
                self.att_probe_criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weight
                )
                self.att_probe_train_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_train_audio_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_train_video_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_audio_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
                self.att_probe_test_video_map = torchmetrics.AveragePrecision(
                    task="multilabel", num_labels=num_classes
                )
            else:
                self.att_probe_criterion = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing
                )
                self.att_probe_train_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_train_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_train_audio_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_train_audio_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_train_video_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_train_video_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_audio_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_audio_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )
                self.att_probe_test_video_acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.att_probe_test_video_acc5 = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                )

    def on_train_epoch_start(self):
        if self.freeze_epochs > 0:
            if self.current_epoch < self.freeze_epochs:
                self.encoder.requires_grad_(False)
            elif self.current_epoch == self.freeze_epochs:
                self.encoder.requires_grad_(True)
                print(f"Epoch {self.current_epoch}: unfreezing encoder")

    def training_step(self, batch, batch_idx):
        video = batch["global_video"][:, 0]
        audio = batch["global_spectrogram"][:, 0]
        targets = batch["label"]

        if self.training and self.modality_drop_prob > 0.0:
            B = video.size(0)
            rolls = torch.rand(B, device=video.device)
            half_p = self.modality_drop_prob / 2
            drop_video = rolls < half_p
            drop_audio = (rolls >= half_p) & (rolls < self.modality_drop_prob)
            if drop_video.any():
                video = video.clone()
                video[drop_video] = 0.0
            if drop_audio.any():
                audio = audio.clone()
                audio[drop_audio] = 0.0

        mixed = False
        if self.mixup_alpha > 0.0 and self.training:
            mix_lam = float(
                torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
                .sample()
                .item()
            )
            mix_perm = torch.randperm(video.size(0), device=video.device)
            video = mix_lam * video + (1.0 - mix_lam) * video[mix_perm]
            audio = mix_lam * audio + (1.0 - mix_lam) * audio[mix_perm]
            if self.multi_label:
                targets = mix_lam * targets + (1.0 - mix_lam) * targets[mix_perm]
            mixed = True

        if self.attentive_probe or self.mean_pool:
            cls_tokens, patch_tokens = self.encoder(video, audio, return_patches=True)
        else:
            cls_tokens = self.encoder(video, audio)

        if self.mean_pool:
            features = patch_tokens.mean(dim=1)
        else:
            features = cls_tokens

        logits = self.head(self.norm(features))
        if mixed and not self.multi_label:
            loss = mix_lam * self.criterion(logits, targets) + (
                1.0 - mix_lam
            ) * self.criterion(logits, targets[mix_perm])
        else:
            loss = self.criterion(logits, targets)

        if not mixed:
            if self.multi_label:
                self.train_map(logits, targets.long())
            else:
                self.train_acc(logits.argmax(dim=1), targets)
                self.train_acc5(logits, targets)

        if self.attentive_probe:
            att_logits = self.att_probe(patch_tokens)
            if mixed and not self.multi_label:
                att_loss = mix_lam * self.att_probe_criterion(att_logits, targets) + (
                    1.0 - mix_lam
                ) * self.att_probe_criterion(att_logits, targets[mix_perm])
            else:
                att_loss = self.att_probe_criterion(att_logits, targets)
            loss = loss + att_loss
            self.log("train/att_probe_loss", att_loss, sync_dist=True)
            if not mixed:
                if self.multi_label:
                    self.att_probe_train_map(att_logits, targets.long())
                    self.log("train/att_map", self.att_probe_train_map, sync_dist=True)
                else:
                    self.att_probe_train_acc(att_logits.argmax(dim=1), targets)
                    self.att_probe_train_acc5(att_logits, targets)
                    self.log("train/att_acc", self.att_probe_train_acc, sync_dist=True)
                    self.log(
                        "train/att_acc_top5", self.att_probe_train_acc5, sync_dist=True
                    )

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        if not mixed:
            if self.multi_label:
                self.log("train/map", self.train_map, sync_dist=True)
            else:
                self.log("train/acc", self.train_acc, sync_dist=True)
                self.log("train/acc_top5", self.train_acc5, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        if not mixed:
            with torch.no_grad():
                video_zero = torch.zeros_like(video)
                audio_zero = torch.zeros_like(audio)
                if self.attentive_probe or self.mean_pool:
                    audio_cls, audio_patches = self.encoder(
                        video_zero, audio, return_patches=True
                    )
                    video_cls, video_patches = self.encoder(
                        video, audio_zero, return_patches=True
                    )
                else:
                    audio_cls = self.encoder(video_zero, audio)
                    video_cls = self.encoder(video, audio_zero)

                if self.mean_pool:
                    audio_features = audio_patches.mean(dim=1)
                    video_features = video_patches.mean(dim=1)
                else:
                    audio_features = audio_cls
                    video_features = video_cls

                audio_logits = self.head(self.norm(audio_features))
                video_logits = self.head(self.norm(video_features))

            if self.multi_label:
                self.train_audio_map(audio_logits, targets.long())
                self.train_video_map(video_logits, targets.long())
                self.log("train/audio_map", self.train_audio_map, sync_dist=True)
                self.log("train/video_map", self.train_video_map, sync_dist=True)
            else:
                self.train_audio_acc(audio_logits.argmax(dim=1), targets)
                self.train_audio_acc5(audio_logits, targets)
                self.train_video_acc(video_logits.argmax(dim=1), targets)
                self.train_video_acc5(video_logits, targets)
                self.log("train/audio_acc", self.train_audio_acc, sync_dist=True)
                self.log("train/audio_acc_top5", self.train_audio_acc5, sync_dist=True)
                self.log("train/video_acc", self.train_video_acc, sync_dist=True)
                self.log("train/video_acc_top5", self.train_video_acc5, sync_dist=True)

            if self.attentive_probe:
                with torch.no_grad():
                    att_audio_logits = self.att_probe(
                        modality_tokens(audio_patches, self.num_video_tokens, "audio")
                    )
                    att_video_logits = self.att_probe(
                        modality_tokens(video_patches, self.num_video_tokens, "video")
                    )
                if self.multi_label:
                    self.att_probe_train_audio_map(att_audio_logits, targets.long())
                    self.att_probe_train_video_map(att_video_logits, targets.long())
                    self.log(
                        "train/att_audio_map",
                        self.att_probe_train_audio_map,
                        sync_dist=True,
                    )
                    self.log(
                        "train/att_video_map",
                        self.att_probe_train_video_map,
                        sync_dist=True,
                    )
                else:
                    self.att_probe_train_audio_acc(
                        att_audio_logits.argmax(dim=1), targets
                    )
                    self.att_probe_train_audio_acc5(att_audio_logits, targets)
                    self.att_probe_train_video_acc(
                        att_video_logits.argmax(dim=1), targets
                    )
                    self.att_probe_train_video_acc5(att_video_logits, targets)
                    self.log(
                        "train/att_audio_acc",
                        self.att_probe_train_audio_acc,
                        sync_dist=True,
                    )
                    self.log(
                        "train/att_audio_acc_top5",
                        self.att_probe_train_audio_acc5,
                        sync_dist=True,
                    )
                    self.log(
                        "train/att_video_acc",
                        self.att_probe_train_video_acc,
                        sync_dist=True,
                    )
                    self.log(
                        "train/att_video_acc_top5",
                        self.att_probe_train_video_acc5,
                        sync_dist=True,
                    )

        return loss

    def test_step(self, batch, batch_idx):
        video = batch["video"]
        audio = batch["spectrogram"]
        targets = batch["label"]
        B, N = video.shape[:2]
        video = video.view(B * N, *video.shape[2:])
        audio = audio.view(B * N, *audio.shape[2:])

        with torch.no_grad():
            if self.attentive_probe or self.mean_pool:
                cls_tokens, patch_tokens = self.encoder(
                    video, audio, return_patches=True
                )
            else:
                cls_tokens = self.encoder(video, audio)

            if self.mean_pool:
                features = patch_tokens.mean(dim=1)
            else:
                features = cls_tokens

            logits = self.head(self.norm(features))
            logits = logits.view(B, N, -1).mean(dim=1)

        self.log(
            "test/loss",
            self.criterion(logits, targets),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self.multi_label:
            self.test_map(logits, targets.long())
            self.log(
                "test/map", self.test_map, on_step=False, on_epoch=True, sync_dist=True
            )
        else:
            self.test_acc(logits.argmax(dim=1), targets)
            self.test_acc5(logits, targets)
            self.log(
                "test/acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True
            )
            self.log(
                "test/acc_top5",
                self.test_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.attentive_probe:
            with torch.no_grad():
                att_logits = self.att_probe(patch_tokens).view(B, N, -1).mean(dim=1)
            if self.multi_label:
                self.att_probe_test_map(att_logits, targets.long())
                self.log(
                    "test/att_map",
                    self.att_probe_test_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.att_probe_test_acc(att_logits.argmax(dim=1), targets)
                self.att_probe_test_acc5(att_logits, targets)
                self.log(
                    "test/att_acc",
                    self.att_probe_test_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/att_acc_top5",
                    self.att_probe_test_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        with torch.no_grad():
            video_zero = torch.zeros_like(video)
            audio_zero = torch.zeros_like(audio)

            if self.attentive_probe or self.mean_pool:
                audio_cls, audio_patches = self.encoder(
                    video_zero, audio, return_patches=True
                )
                video_cls, video_patches = self.encoder(
                    video, audio_zero, return_patches=True
                )
            else:
                audio_cls = self.encoder(video_zero, audio)
                video_cls = self.encoder(video, audio_zero)

            if self.mean_pool:
                audio_features = audio_patches.mean(dim=1)
                video_features = video_patches.mean(dim=1)
            else:
                audio_features = audio_cls
                video_features = video_cls

            audio_logits = (
                self.head(self.norm(audio_features)).view(B, N, -1).mean(dim=1)
            )
            video_logits = (
                self.head(self.norm(video_features)).view(B, N, -1).mean(dim=1)
            )

        if self.multi_label:
            self.test_audio_map(audio_logits, targets.long())
            self.test_video_map(video_logits, targets.long())
            self.log(
                "test/audio_map",
                self.test_audio_map,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test/video_map",
                self.test_video_map,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            self.test_audio_acc(audio_logits.argmax(dim=1), targets)
            self.test_audio_acc5(audio_logits, targets)
            self.test_video_acc(video_logits.argmax(dim=1), targets)
            self.test_video_acc5(video_logits, targets)
            self.log(
                "test/audio_acc",
                self.test_audio_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test/audio_acc_top5",
                self.test_audio_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test/video_acc",
                self.test_video_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test/video_acc_top5",
                self.test_video_acc5,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.attentive_probe:
            att_audio_logits = (
                self.att_probe(
                    modality_tokens(audio_patches, self.num_video_tokens, "audio")
                )
                .view(B, N, -1)
                .mean(dim=1)
            )
            att_video_logits = (
                self.att_probe(
                    modality_tokens(video_patches, self.num_video_tokens, "video")
                )
                .view(B, N, -1)
                .mean(dim=1)
            )
            if self.multi_label:
                self.att_probe_test_audio_map(att_audio_logits, targets.long())
                self.att_probe_test_video_map(att_video_logits, targets.long())
                self.log(
                    "test/att_audio_map",
                    self.att_probe_test_audio_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/att_video_map",
                    self.att_probe_test_video_map,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.att_probe_test_audio_acc(att_audio_logits.argmax(dim=1), targets)
                self.att_probe_test_audio_acc5(att_audio_logits, targets)
                self.att_probe_test_video_acc(att_video_logits.argmax(dim=1), targets)
                self.att_probe_test_video_acc5(att_video_logits, targets)
                self.log(
                    "test/att_audio_acc",
                    self.att_probe_test_audio_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/att_audio_acc_top5",
                    self.att_probe_test_audio_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/att_video_acc",
                    self.att_probe_test_video_acc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/att_video_acc_top5",
                    self.att_probe_test_video_acc5,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def configure_optimizers(self):
        head_params = list(self.norm.parameters()) + list(self.head.parameters())
        if self.attentive_probe:
            head_params += list(self.att_probe.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.encoder.parameters() if id(p) not in head_ids
        ]

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.lr * self.backbone_lr_scale,
                "weight_decay": self.weight_decay,
            },
            {"params": head_params, "lr": self.lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(param_groups)

        accumulation_steps = self.trainer.accumulate_grad_batches or 1
        effective_batch_size = (
            self.hparams.batch_size * self.trainer.world_size * accumulation_steps
        )
        steps_per_epoch = self.total_samples // effective_batch_size
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_fraction)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
