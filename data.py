import os
import webdataset as wds
import pandas as pd

import torch
import torchaudio
from torch import nn

from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchcodec.decoders import VideoDecoder, AudioDecoder
import torchaudio.transforms as T


class RandomTubeMasking(nn.Module):
    def __init__(self, patch_size=(2, 16, 16), mask_ratio=0.6, mask_val=0.0):
        super().__init__()
        self.patch_t, self.patch_h, self.patch_w = patch_size
        self.mask_ratio = mask_ratio
        self.mask_val = mask_val

    def forward(self, x):
        single = x.dim() == 4
        if single:
            x = x.unsqueeze(0)

        B, C, T, H, W = x.shape
        grid_t = T // self.patch_t
        grid_h = H // self.patch_h
        grid_w = W // self.patch_w
        num_patches = grid_t * grid_h * grid_w
        num_masked = int(num_patches * self.mask_ratio)

        noise = torch.rand(B, num_patches, device=x.device)
        ids_sorted = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, num_patches, device=x.device)
        mask.scatter_(1, ids_sorted[:, :num_masked], 1.0)

        mask = mask.view(B, grid_t, grid_h, grid_w)
        mask = (
            mask.repeat_interleave(self.patch_t, dim=1)
            .repeat_interleave(self.patch_h, dim=2)
            .repeat_interleave(self.patch_w, dim=3)
        )

        if mask.shape[1:] != (T, H, W):
            pad_t = T - mask.shape[1]
            pad_h = H - mask.shape[2]
            pad_w = W - mask.shape[3]
            mask = torch.nn.functional.pad(
                mask, (0, pad_w, 0, pad_h, 0, pad_t), value=0
            )

        mask = mask.unsqueeze(1).expand_as(x)
        result = x * (1 - mask) + (mask * self.mask_val)

        if single:
            result = result.squeeze(0)
        return result


class BlockTubeMasking(nn.Module):
    """Masks a single contiguous spatial block (h, w) across all time steps.

    Unlike random tube masking, the visible patches form a coherent spatial
    region rather than scattered pixels, making the invariance task harder and
    encouraging the model to reason about spatial context.
    """

    def __init__(self, patch_size=(2, 16, 16), mask_ratio=0.85, mask_val=0.0):
        super().__init__()
        self.patch_t, self.patch_h, self.patch_w = patch_size
        self.mask_ratio = mask_ratio
        self.mask_val = mask_val

    def forward(self, x):
        if self.mask_ratio <= 0:
            return x

        single = x.dim() == 4
        if single:
            x = x.unsqueeze(0)

        B, C, T, H, W = x.shape
        grid_t = T // self.patch_t
        grid_h = H // self.patch_h
        grid_w = W // self.patch_w

        target_area = self.mask_ratio * grid_h * grid_w

        spatial_mask = torch.zeros(B, grid_h, grid_w, device=x.device)
        for b in range(B):
            log_r = torch.empty(1).uniform_(-0.5, 0.5).item()
            r = torch.tensor(log_r).exp().item()
            bh = max(1, min(grid_h, round((target_area * r) ** 0.5)))
            bw = max(1, min(grid_w, round((target_area / r) ** 0.5)))
            h0 = torch.randint(0, max(1, grid_h - bh + 1), (1,)).item()
            w0 = torch.randint(0, max(1, grid_w - bw + 1), (1,)).item()
            spatial_mask[b, h0:h0 + bh, w0:w0 + bw] = 1.0

        mask = spatial_mask.unsqueeze(1).expand(B, grid_t, grid_h, grid_w).contiguous()
        mask = (
            mask.repeat_interleave(self.patch_t, dim=1)
            .repeat_interleave(self.patch_h, dim=2)
            .repeat_interleave(self.patch_w, dim=3)
        )

        if mask.shape[1:] != (T, H, W):
            pad_t = T - mask.shape[1]
            pad_h = H - mask.shape[2]
            pad_w = W - mask.shape[3]
            mask = torch.nn.functional.pad(
                mask, (0, pad_w, 0, pad_h, 0, pad_t), value=0
            )

        mask = mask.unsqueeze(1).expand_as(x)
        result = x * (1 - mask) + (mask * self.mask_val)

        if single:
            result = result.squeeze(0)
        return result


SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400
N_MELS = 128

VGGSOUND_SPEC_MEAN = -20.437003
VGGSOUND_SPEC_STD = 24.496246


class VideoAudioPipeline:
    def __init__(
        self,
        label_csv_path,
        is_train=True,
        debug=False,
        frame_size=(224, 224),
        num_frames=8,
        num_global_views=2,
        num_local_views=4,
        num_eval_clips=4,
        classes=None,
        video_mask_ratio=0.80,
        freq_mask_param=64,
        time_mask_param=256,
        modality_drop_prob=0.5,
        color_jitter=0.0,
        gaussian_blur=0.0,
        random_grayscale=0.0,
        solarize=0.0,
        audio_noise=0.0,
        audio_gain=0.0,
        spec_aug_global=False,
        global_rrc_min_scale=0.0,
        video_token_crop=False,
        crop_scale=0.4,
        csv_format="vggsound",
        spec_mean=VGGSOUND_SPEC_MEAN,
        spec_std=VGGSOUND_SPEC_STD,
    ):
        self.is_train = is_train
        self.debug = debug
        self.sample_rate = SAMPLE_RATE
        self.target_duration = 8
        self.fps = 25
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.num_eval_clips = num_eval_clips
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.csv_format = csv_format

        self.audio_len = self.sample_rate * self.target_duration
        self.classes = []
        if csv_format == "audioset":
            self.labels_map = self._load_audioset_labels_map(label_csv_path, classes=classes)
        else:
            self.labels_map = self._load_labels_map(label_csv_path, classes=classes)

        self.spectrogram_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=N_MELS,
                n_fft=N_FFT,
                win_length=N_FFT,
                hop_length=HOP_LENGTH,
                window_fn=torch.hamming_window,
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )

        if global_rrc_min_scale > 0:
            global_aug_list = [
                v2.RandomResizedCrop(frame_size, scale=(global_rrc_min_scale, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
            ]
        else:
            global_aug_list = [
                v2.Resize(256, antialias=True),
                v2.CenterCrop(frame_size),
                v2.RandomHorizontalFlip(),
            ]
        if color_jitter > 0:
            s = color_jitter
            global_aug_list.append(
                v2.RandomApply([v2.ColorJitter(0.2 * s, 0.2 * s, 0.1 * s, 0.0)], p=0.8)
            )
        if gaussian_blur > 0:
            global_aug_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=gaussian_blur * 0.2)
            )
        if solarize > 0:
            global_aug_list.append(v2.RandomSolarize(threshold=128, p=solarize))
        global_aug_list.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.global_video_transform = v2.Compose(global_aug_list)

        local_aug_list = [
            v2.RandomResizedCrop(frame_size, scale=(0.4, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
        ]
        if color_jitter > 0:
            s = color_jitter
            local_aug_list.append(
                v2.RandomApply([v2.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)], p=0.8)
            )
        if gaussian_blur > 0:
            local_aug_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=gaussian_blur)
            )
        if random_grayscale > 0:
            local_aug_list.append(v2.RandomGrayscale(p=random_grayscale))
        local_aug_list.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.local_video_transform = v2.Compose(local_aug_list)

        self.val_video_transform = v2.Compose(
            [
                v2.Resize(frame_size, antialias=True),
                v2.CenterCrop(frame_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.audio_aug = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=freq_mask_param),
            T.TimeMasking(time_mask_param=time_mask_param),
        )
        self.video_mask = BlockTubeMasking(
            patch_size=(2, 16, 16), mask_ratio=video_mask_ratio
        )
        self.modality_drop_prob = modality_drop_prob
        self.audio_noise = audio_noise
        self.audio_gain = audio_gain
        self.spec_aug_global = spec_aug_global
        self.clean_survivor = False
        self.cross_modal = False
        self.mask_cross_modal = False

        self.video_token_crop = video_token_crop
        self.crop_scale = crop_scale
        patch = 16
        self.crop_grid = frame_size[0] // patch
        self.crop_side = max(
            1, min(self.crop_grid, round((crop_scale ** 0.5) * self.crop_grid))
        )
        self.crop_n_keep = self.crop_side * self.crop_side

        local_full_list = [
            v2.Resize(256, antialias=True),
            v2.CenterCrop(frame_size),
            v2.RandomHorizontalFlip(),
        ]
        if color_jitter > 0:
            s = color_jitter
            local_full_list.append(
                v2.RandomApply([v2.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)], p=0.8)
            )
        if gaussian_blur > 0:
            local_full_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=gaussian_blur)
            )
        if random_grayscale > 0:
            local_full_list.append(v2.RandomGrayscale(p=random_grayscale))
        local_full_list.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.local_fullframe_transform = v2.Compose(local_full_list)

        self._resamplers = {}

    def _sample_spatial_keep_idx(self):
        """Sample a contiguous square block of spatial patches at a random pos.

        Returns a [crop_n_keep] long tensor of indices into the row-major
        (crop_grid x crop_grid) spatial patch grid. The block size is fixed so
        the kept-token count is constant across views/samples, keeping the
        batched local-view tensor stackable.
        """
        grid, side = self.crop_grid, self.crop_side
        h0 = torch.randint(0, grid - side + 1, (1,)).item() if grid > side else 0
        w0 = torch.randint(0, grid - side + 1, (1,)).item() if grid > side else 0
        rows = torch.arange(h0, h0 + side)
        cols = torch.arange(w0, w0 + side)
        return (rows[:, None] * grid + cols[None, :]).reshape(-1)

    def _process_local_token_crop(self, local_clips, local_starts, waveform, local_aug_fn):
        """AV-JEPA local views: full-frame video (token-cropped in the model),
        independent per-modality dropout with a no-empty guarantee, and a
        per-view spatial bounding box returned as patch-token keep indices."""
        K = len(local_clips)
        videos, specs = [], []
        for raw_video, si in zip(local_clips, local_starts):
            lv = self.local_fullframe_transform(raw_video)
            lv = lv.permute(1, 0, 2, 3)
            videos.append(lv)

            audio_clip = self._slice_audio(waveform, si)
            _, spec = self._make_spectrogram(audio_clip, augment_fn=local_aug_fn)
            specs.append(spec)
        videos = torch.stack(videos)
        specs = torch.stack(specs)

        p = self.modality_drop_prob
        drop_audio = torch.rand(K) < p
        drop_video = torch.rand(K) < p
        both = drop_audio & drop_video
        keep_audio = torch.rand(K) < 0.5
        drop_audio = drop_audio & ~(both & keep_audio)
        drop_video = drop_video & ~(both & ~keep_audio)
        for k in range(K):
            if drop_video[k]:
                videos[k] = 0.0
            if drop_audio[k]:
                specs[k] = 0.0

        keep_idx = torch.stack([self._sample_spatial_keep_idx() for _ in range(K)])
        return videos, specs, keep_idx

    def _load_labels_map(self, label_csv_path, classes=None):
        df = pd.read_csv(label_csv_path, header=None, names=["filename", "label"])
        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted(df["label"].unique())
        label_to_idx = {label: i for i, label in enumerate(self.classes)}

        keys = df["filename"].astype(str).apply(lambda x: os.path.splitext(x)[0])
        labels = df["label"].map(label_to_idx)
        return dict(zip(keys, labels))

    def _load_audioset_labels_map(self, csv_path, classes=None):
        """Load AudioSet multi-label annotations from segments CSV."""
        rows = []
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(", ", 3)
                ytid = parts[0]
                start_ms = round(float(parts[1]) * 1000)
                end_ms = round(float(parts[2]) * 1000)
                label_ids = parts[3].strip('"').split(",")
                key = f"{ytid}_{start_ms}_{end_ms}"
                rows.append((key, label_ids))

        if classes is not None:
            self.classes = classes
        else:
            all_ids = set()
            for _, label_ids in rows:
                all_ids.update(label_ids)
            self.classes = sorted(all_ids)

        label_to_idx = {label: i for i, label in enumerate(self.classes)}
        num_classes = len(self.classes)

        labels_map = {}
        for key, label_ids in rows:
            multi_hot = torch.zeros(num_classes, dtype=torch.float32)
            for lid in label_ids:
                if lid in label_to_idx:
                    multi_hot[label_to_idx[lid]] = 1.0
            labels_map[key] = multi_hot
        return labels_map

    def _get_normalized_key(self, sample):
        raw_key = sample["__key__"]
        return os.path.basename(raw_key)

    def has_label(self, sample):
        key = self._get_normalized_key(sample)
        return key in self.labels_map

    def _decode_clips_batched(self, video_decoder, total_frames, stride, start_indices):
        """Decode all clips in a single batched get_frames_at call."""
        all_indices = []
        clip_lengths = []
        for si in start_indices:
            ideal = si + torch.arange(self.num_frames) * stride
            valid = ideal[ideal < total_frames]
            all_indices.append(valid)
            clip_lengths.append(len(valid))

        all_indices_cat = torch.cat(all_indices)
        all_frames = video_decoder.get_frames_at(indices=all_indices_cat).data

        clips = []
        offset = 0
        for length in clip_lengths:
            clip = all_frames[offset : offset + length]
            frames_needed = self.num_frames - clip.shape[0]
            if frames_needed > 0:
                padding = torch.zeros(frames_needed, *clip.shape[1:], dtype=clip.dtype)
                clip = torch.cat([clip, padding], dim=0)
            clips.append(clip)
            offset += length
        return clips

    def _resample_waveform(self, waveform, original_sample_rate):
        """Resample the full waveform once to the target sample rate."""
        if original_sample_rate != self.sample_rate:
            if original_sample_rate not in self._resamplers:
                self._resamplers[original_sample_rate] = torchaudio.transforms.Resample(
                    original_sample_rate, self.sample_rate
                )
            waveform = self._resamplers[original_sample_rate](waveform)
        return waveform

    def _slice_audio(self, waveform, start_idx):
        """Slice pre-resampled audio to match a video clip starting at start_idx."""
        start_sample = int((start_idx / self.fps) * self.sample_rate)
        end_sample = start_sample + self.audio_len

        if waveform.shape[1] >= end_sample:
            clip = waveform[:, start_sample:end_sample]
        else:
            clip = waveform[:, start_sample:]
            if clip.shape[1] < self.audio_len:
                clip = torch.nn.functional.pad(
                    clip, (0, self.audio_len - clip.shape[1])
                )
            clip = clip[:, : self.audio_len]
        return clip

    def _augment_spectrogram(self, spec, is_local=False):
        """Apply spectral augmentations to a spectrogram (before z-norm)."""
        if self.audio_gain > 0:
            max_gain = self.audio_gain if is_local else self.audio_gain * 0.4
            gain = torch.empty(1).uniform_(-max_gain, max_gain).item()
            spec = spec + gain
        if is_local and self.audio_noise > 0:
            noise = torch.randn_like(spec) * self.audio_noise
            spec = spec + noise
        return spec

    def _make_spectrogram(self, audio, augment_fn=None):
        """Pad/truncate audio and compute z-normalised mel spectrogram."""
        current_len = audio.shape[1]
        if current_len > self.audio_len:
            audio = audio[:, : self.audio_len]
        elif current_len < self.audio_len:
            pad_amount = self.audio_len - current_len
            audio = torch.nn.functional.pad(audio, (0, pad_amount))

        spec = self.spectrogram_transform(audio)
        if augment_fn is not None:
            spec = augment_fn(spec)
        spec = (spec - self.spec_mean) / self.spec_std
        return audio, spec

    @torch.no_grad()
    def process(self, sample):
        """Decode and process a sample in one pass (merged decode + transform)."""
        key = self._get_normalized_key(sample)
        if key not in self.labels_map:
            return None

        video_bytes = sample.get("mp4")
        if video_bytes is None:
            return None

        try:
            video_decoder = VideoDecoder(video_bytes, device="cpu")
            total_frames = len(video_decoder)
            stride = int((self.target_duration * self.fps) / self.num_frames)
            max_start_video = max(0, total_frames - (stride * self.num_frames))

            audio_decoder = AudioDecoder(video_bytes)
            audio_samples = audio_decoder.get_all_samples()
            waveform = audio_samples.data
            original_sample_rate = audio_samples.sample_rate
            del audio_decoder, audio_samples

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = self._resample_waveform(waveform, original_sample_rate)

            max_start_audio = max(
                0,
                int((waveform.shape[1] - self.audio_len) / self.sample_rate * self.fps),
            )
            max_start = min(max_start_video, max_start_audio)
            label = self.labels_map[key]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)

            if self.is_train:
                result = self._process_train(
                    video_decoder, total_frames, stride, max_start, waveform, label
                )
            else:
                result = self._process_val(
                    video_decoder, total_frames, stride, max_start, waveform, label
                )
            del video_decoder
            return result
        except Exception as e:
            print(f"Skipping corrupt sample {sample.get('__key__', '?')}: {e}")
            return None

    def _process_train(
        self, video_decoder, total_frames, stride, max_start, waveform, label
    ):
        n_total = self.num_global_views + self.num_local_views
        if max_start > 0:
            all_start_indices = torch.randint(0, max_start + 1, (n_total,)).tolist()
        else:
            all_start_indices = [0] * n_total

        global_starts = all_start_indices[: self.num_global_views]
        local_starts  = all_start_indices[self.num_global_views :]

        decoded_clips = self._decode_clips_batched(
            video_decoder, total_frames, stride, all_start_indices
        )
        global_clips = decoded_clips[: self.num_global_views]
        local_clips  = decoded_clips[self.num_global_views :]

        if self.num_global_views > 0:
            global_videos = []
            global_specs = []
            global_aug_fn = lambda s: self._augment_spectrogram(s, is_local=False)
            for raw_video, si in zip(global_clips, global_starts):
                gv = self.global_video_transform(raw_video)
                gv = gv.permute(1, 0, 2, 3)
                global_videos.append(gv)

                audio_clip = self._slice_audio(waveform, si)
                _, spec = self._make_spectrogram(audio_clip, augment_fn=global_aug_fn)
                if self.spec_aug_global:
                    spec = self.audio_aug(spec)
                global_specs.append(spec)

            global_videos = torch.stack(global_videos)
            global_specs = torch.stack(global_specs)
        else:
            global_videos = torch.empty(0)
            global_specs = torch.empty(0)

        local_aug_fn = lambda s: self._augment_spectrogram(s, is_local=True)
        local_keep_idx = None
        if self.video_token_crop:
            local_videos, local_specs, local_keep_idx = self._process_local_token_crop(
                local_clips, local_starts, waveform, local_aug_fn
            )
        elif self.cross_modal:
            local_video_list = []
            local_spec_list = []
            for idx, (raw_video, si) in enumerate(zip(local_clips, local_starts)):
                lv = self.local_video_transform(raw_video)
                lv = lv.permute(1, 0, 2, 3)

                audio_clip = self._slice_audio(waveform, si)
                _, spec = self._make_spectrogram(audio_clip, augment_fn=local_aug_fn)

                if idx % 2 == 0:
                    spec_a = self.audio_aug(spec) if self.mask_cross_modal else spec
                    local_video_list.append(torch.zeros_like(lv))
                    local_spec_list.append(spec_a)
                else:
                    if self.mask_cross_modal:
                        lv_v = self.video_mask(lv.unsqueeze(0)).squeeze(0)
                    else:
                        lv_v = lv
                    local_video_list.append(lv_v)
                    local_spec_list.append(torch.zeros_like(spec))

            local_videos = torch.stack(local_video_list)
            local_specs = torch.stack(local_spec_list)
        else:
            local_views = []
            local_spec_list = []
            for raw_video, si in zip(local_clips, local_starts):
                lv = self.local_video_transform(raw_video)
                lv = lv.permute(1, 0, 2, 3)
                local_views.append(lv)

                audio_clip = self._slice_audio(waveform, si)
                _, spec = self._make_spectrogram(audio_clip, augment_fn=local_aug_fn)
                local_spec_list.append(self.audio_aug(spec))

            lv = torch.stack(local_views)
            local_specs_raw = torch.stack(local_spec_list)

            if self.modality_drop_prob > 0:
                K = lv.shape[0]
                rolls = torch.rand(K)
                half_p = self.modality_drop_prob / 2
                drop_video = rolls < half_p
                drop_audio = (rolls >= half_p) & (rolls < self.modality_drop_prob)

                if self.clean_survivor and drop_video.any():
                    local_specs_raw[drop_video] = torch.stack(
                        [self._make_spectrogram(self._slice_audio(waveform, local_starts[k]))[1]
                         for k in range(K) if drop_video[k]]
                    )

                local_videos = self.video_mask(lv)
                local_specs = local_specs_raw.clone()

                if self.clean_survivor and drop_audio.any():
                    local_videos[drop_audio] = lv[drop_audio]

                for k in range(K):
                    if drop_video[k]:
                        local_videos[k] = 0.0
                    elif drop_audio[k]:
                        local_specs[k] = 0.0
            else:
                local_videos = self.video_mask(lv)
                local_specs = local_specs_raw

        result = {
            "global_video": global_videos,
            "global_spectrogram": global_specs,
            "local_video": local_videos,
            "local_spectrogram": local_specs,
            "label": label,
        }
        if local_keep_idx is not None:
            result["local_video_keep_idx"] = local_keep_idx
        return result

    def _process_val(
        self, video_decoder, total_frames, stride, max_start, waveform, label
    ):
        if max_start > 0:
            start_indices = torch.randint(
                0, max_start + 1, (self.num_eval_clips,)
            ).tolist()
        else:
            start_indices = [0] * self.num_eval_clips

        decoded_clips = self._decode_clips_batched(
            video_decoder, total_frames, stride, start_indices
        )

        videos, specs, waveforms = [], [], []
        for raw_v, si in zip(decoded_clips, start_indices):
            v = self.val_video_transform(raw_v)
            v = v.permute(1, 0, 2, 3)
            videos.append(v)

            audio_clip = self._slice_audio(waveform, si)
            a, s = self._make_spectrogram(audio_clip)
            specs.append(s)
            waveforms.append(a)

        result = {
            "video": torch.stack(videos),
            "spectrogram": torch.stack(specs),
            "label": label,
        }
        if not self.debug:
            result["waveform"] = torch.stack(waveforms)
        return result


def compute_audioset_pos_weight(csv_path, classes):
    """Compute per-class pos_weight = num_negatives / num_positives for BCEWithLogitsLoss."""
    label_to_idx = {label: i for i, label in enumerate(classes)}
    num_classes = len(classes)
    pos_counts = torch.zeros(num_classes, dtype=torch.float32)
    n_samples = 0
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(", ", 3)
            label_ids = parts[3].strip('"').split(",")
            for lid in label_ids:
                if lid in label_to_idx:
                    pos_counts[label_to_idx[lid]] += 1.0
            n_samples += 1
    neg_counts = n_samples - pos_counts
    return neg_counts / pos_counts.clamp(min=1.0)


def get_dataloader(
    tar_path,
    csv_path,
    test_tar_path=None,
    test_csv_path=None,
    debug=False,
    batch_size=64,
    num_workers=2,
    num_workers_test=2,
    frame_size=(224, 224),
    num_frames=8,
    num_global_views=2,
    num_local_views=4,
    num_eval_clips=4,
    train_size=None,
    test_size=None,
    video_mask_ratio=0.80,
    freq_mask_param=64,
    time_mask_param=256,
    modality_drop_prob=0.5,
    clean_survivor=False,
    cross_modal=False,
    mask_cross_modal=False,
    world_size=1,
    color_jitter=0.0,
    gaussian_blur=0.0,
    random_grayscale=0.0,
    solarize=0.0,
    audio_noise=0.0,
    audio_gain=0.0,
    spec_aug_global=False,
    global_rrc_min_scale=0.0,
    video_token_crop=False,
    crop_scale=0.4,
    csv_format="vggsound",
    spec_mean=VGGSOUND_SPEC_MEAN,
    spec_std=VGGSOUND_SPEC_STD,
):
    def create_dataset_and_loader(tars, pipeline, is_train=True, num_samples=None):
        dataset = wds.WebDataset(
            tars,
            shardshuffle=100 if is_train else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            empty_check=True if is_train else False
        )
        if is_train:
            dataset = dataset.shuffle(400)

        dataset = (
            dataset.select(pipeline.has_label)
            .map(pipeline.process)
            .select(lambda x: x is not None)
            .batched(batch_size)
        )

        if num_samples is not None:
            num_batches = int(num_samples // batch_size // world_size)
            dataset = dataset.with_epoch(num_batches).with_length(num_batches)

        n_workers = num_workers if is_train else num_workers_test

        loader_kwargs = {
            "batch_size": None,
            "num_workers": n_workers,
            "persistent_workers": False,
            "pin_memory": True,
        }
        if n_workers > 0:
            loader_kwargs["prefetch_factor"] = 3

        loader = DataLoader(dataset, **loader_kwargs)
        return loader
    
    print("Data Parameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Number of Workers: {num_workers}")
    print(f"  Number of Test Workers: {num_workers_test}")
    print(f"  Frame Size: {frame_size}")
    print(f"  Number of Frames: {num_frames}")
    print(f"  Number of Global Views: {num_global_views}")
    print(f"  Number of Local Views: {num_local_views}")
    print(f"  Number of Evaluation Clips: {num_eval_clips}")
    print(f"  Video Mask Ratio: {video_mask_ratio}")
    print(f"  Frequency Mask Parameter: {freq_mask_param}/128")
    print(f"  Time Mask Parameter: {time_mask_param}/801")
    print(f"  Modality Drop Probability: {modality_drop_prob}")
    print(f"  Clean Survivor: {clean_survivor}")
    print(f"  Cross-Modal Mode: {cross_modal}")
    print(f"  Mask Cross-Modal Locals: {mask_cross_modal}")
    print(f"  Color Jitter: {color_jitter}")
    print(f"  Gaussian Blur: {gaussian_blur}")
    print(f"  Random Grayscale: {random_grayscale}")
    print(f"  Solarize: {solarize}")
    print(f"  Audio Noise: {audio_noise}")
    print(f"  Audio Gain: {audio_gain}")
    print(f"  SpecAugment on Global Views: {spec_aug_global}")
    print(f"  Global RandomResizedCrop Min Scale: {global_rrc_min_scale}")
    print(f"  Video Token Crop (AV-JEPA): {video_token_crop}")
    print(f"  Token Crop Scale: {crop_scale}")

    train_pipeline = VideoAudioPipeline(
        csv_path,
        is_train=True,
        debug=debug,
        frame_size=frame_size,
        num_frames=num_frames,
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        video_mask_ratio=video_mask_ratio,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        modality_drop_prob=modality_drop_prob,
        color_jitter=color_jitter,
        gaussian_blur=gaussian_blur,
        random_grayscale=random_grayscale,
        solarize=solarize,
        audio_noise=audio_noise,
        audio_gain=audio_gain,
        spec_aug_global=spec_aug_global,
        global_rrc_min_scale=global_rrc_min_scale,
        video_token_crop=video_token_crop,
        crop_scale=crop_scale,
        csv_format=csv_format,
        spec_mean=spec_mean,
        spec_std=spec_std,
    )
    train_pipeline.clean_survivor = clean_survivor
    train_pipeline.cross_modal = cross_modal
    train_pipeline.mask_cross_modal = mask_cross_modal
    train_loader = create_dataset_and_loader(
        tar_path, train_pipeline, is_train=True, num_samples=train_size
    )

    if test_tar_path and test_csv_path:
        test_pipeline = VideoAudioPipeline(
            test_csv_path,
            is_train=False,
            debug=debug,
            frame_size=frame_size,
            num_frames=num_frames,
            num_eval_clips=num_eval_clips,
            classes=train_pipeline.classes,
            video_mask_ratio=video_mask_ratio,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            csv_format=csv_format,
            spec_mean=spec_mean,
            spec_std=spec_std,
        )
        test_loader = create_dataset_and_loader(
            test_tar_path, test_pipeline, is_train=False, num_samples=test_size
        )
        return train_loader, test_loader, train_pipeline.classes

    return train_loader, train_pipeline.classes
