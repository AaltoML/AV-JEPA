import torch
import torch.nn as nn

class AudioPatchEmbeddings(nn.Module): 
		def __init__(self, config, t_config):
				super().__init__()
				self.mel_bins, self.time_size = config["spectrogram_size"]
				self.patch_size = config["patch_size"]
				self.patch_stride = config["patch_stride"]
				self.num_channels = config["num_channels"]
				self.hidden_size = t_config["hidden_size"]

				self.num_patches_h = (self.mel_bins - self.patch_size[0] ) // self.patch_stride[0] + 1
				self.num_patches_w = (self.time_size - self.patch_size[1]) // self.patch_stride[1] + 1
				self.num_patches = self.num_patches_h * self.num_patches_w

				self.projection = nn.Conv2d(
						self.num_channels, 
						self.hidden_size, 
						kernel_size=self.patch_size,
						stride=self.patch_stride
				)

		def forward(self, x):
				x = self.projection(x) 
				x = x.flatten(2).transpose(1, 2) 
				return x
		

class VideoPatchEmbeddings(nn.Module): 
	def __init__(self, config, t_config):
		super().__init__()
		self.image_size = config["image_size"]
		self.num_frames = config["num_frames"]
		self.patch_size = config["patch_size"]
		self.tubelet_size = config["tubelet_size"]
		self.num_channels = config["num_channels"]
		self.hidden_size = t_config["hidden_size"]

		num_spatial_patches = (self.image_size // self.patch_size) ** 2
		num_temporal_patches = (self.num_frames // self.tubelet_size)
		self.num_patches = num_spatial_patches * num_temporal_patches

		self.projection = nn.Conv3d(
			self.num_channels, 
			self.hidden_size, 
			kernel_size=(self.tubelet_size, self.patch_size, self.patch_size), 
			stride=(self.tubelet_size, self.patch_size, self.patch_size)
		)

	def forward(self, x):
		x = self.projection(x)
		x = x.flatten(2).transpose(1, 2)
		return x
	

class AudioEmbeddings(nn.Module):
	def __init__(self, a_config, t_config):
		super().__init__()
		self.audio_patch_embed = AudioPatchEmbeddings(a_config, t_config)

		self.audio_freq_pos_embed = nn.Parameter(
			torch.randn(1, self.audio_patch_embed.num_patches_h, 1, t_config["hidden_size"])
		)
		self.audio_time_pos_embed = nn.Parameter(
			torch.randn(1, 1, self.audio_patch_embed.num_patches_w, t_config["hidden_size"])
		)

		self.cls_token = nn.Parameter(torch.randn(1, 1, t_config["hidden_size"]))
		self.layernorm = nn.LayerNorm(t_config["hidden_size"], eps=1e-6)
		self.dropout = nn.Dropout(t_config["hidden_dropout_prob"])

	def forward(self, audio_x):
		batch_size = audio_x.shape[0]
		audio_tokens = self.audio_patch_embed.projection(audio_x)
		audio_tokens = audio_tokens.permute(0, 2, 3, 1)
		audio_tokens = audio_tokens + self.audio_freq_pos_embed + self.audio_time_pos_embed
		audio_tokens = audio_tokens.flatten(1, 2)

		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_tokens, audio_tokens), dim=1)
		x = self.layernorm(x)
		x = self.dropout(x)
		return x


class VideoEmbeddings(nn.Module):
	def __init__(self, v_config, t_config):
		super().__init__()
		self.video_patch_embed = VideoPatchEmbeddings(v_config, t_config)

		self.num_temporal_patches = v_config["num_frames"] // v_config["tubelet_size"]
		self.num_spatial_patches = (v_config["image_size"] // v_config["patch_size"]) ** 2

		self.video_time_embed = nn.Parameter(
			torch.randn(1, self.num_temporal_patches, 1, t_config["hidden_size"])
		)
		self.video_spatial_embed = nn.Parameter(
			torch.randn(1, 1, self.num_spatial_patches, t_config["hidden_size"])
		)

		self.cls_token = nn.Parameter(torch.randn(1, 1, t_config["hidden_size"]))
		self.layernorm = nn.LayerNorm(t_config["hidden_size"], eps=1e-6)
		self.dropout = nn.Dropout(t_config["hidden_dropout_prob"])

	def forward(self, video_x):
		batch_size = video_x.shape[0]
		video_tokens = self.video_patch_embed(video_x)
		video_tokens = video_tokens.reshape(
			batch_size,
			self.num_temporal_patches,
			self.num_spatial_patches,
			-1
		)
		video_tokens = video_tokens + self.video_time_embed + self.video_spatial_embed
		video_tokens = video_tokens.flatten(1, 2)

		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_tokens, video_tokens), dim=1)
		x = self.layernorm(x)
		x = self.dropout(x)
		return x


class EarlyFusionEmbeddings(nn.Module):
	def __init__(self, a_config, v_config, t_config):
		super().__init__()
		
		self.video_patch_embed = VideoPatchEmbeddings(v_config, t_config)
		self.audio_patch_embed = AudioPatchEmbeddings(a_config, t_config)

		self.modality_type_embeddings = nn.Embedding(2, t_config["hidden_size"])

		self.audio_freq_pos_embed = nn.Parameter(
			torch.randn(1, self.audio_patch_embed.num_patches_h, 1, t_config["hidden_size"])
		)
		
		self.audio_time_pos_embed = nn.Parameter(
			torch.randn(1, 1, self.audio_patch_embed.num_patches_w, t_config["hidden_size"])
		)

		self.num_temporal_patches = v_config["num_frames"] // v_config["tubelet_size"]
		self.num_spatial_patches = (v_config["image_size"] // v_config["patch_size"]) ** 2

		self.video_time_embed = nn.Parameter(
			torch.randn(1, self.num_temporal_patches, 1, t_config["hidden_size"])
		)
		self.video_spatial_embed = nn.Parameter(
			torch.randn(1, 1, self.num_spatial_patches, t_config["hidden_size"])
		)
		
		self.cls_token = nn.Parameter(torch.randn(1, 1, t_config["hidden_size"]))
		self.layernorm = nn.LayerNorm(t_config["hidden_size"], eps=1e-6)
		self.dropout = nn.Dropout(t_config["hidden_dropout_prob"])


	def forward(self, video_x, audio_x, video_keep_idx=None):
		batch_size = video_x.shape[0]

		video_tokens = self.video_patch_embed(video_x)

		video_tokens = video_tokens.reshape(
			batch_size,
			self.num_temporal_patches,
			self.num_spatial_patches,
			-1
		)
		video_tokens = video_tokens + self.video_time_embed + self.video_spatial_embed

		if video_keep_idx is not None:
			dim = video_tokens.shape[-1]
			n_keep = video_keep_idx.shape[1]
			idx = video_keep_idx[:, None, :, None].expand(
				batch_size, self.num_temporal_patches, n_keep, dim
			)
			video_tokens = torch.gather(video_tokens, 2, idx)

		video_tokens = video_tokens.flatten(1, 2)
		video_type_embedding = self.modality_type_embeddings(
			torch.zeros(1, 1, device=video_x.device).long()
		)
		video_tokens = video_tokens + video_type_embedding


		audio_tokens = self.audio_patch_embed.projection(audio_x) 
		audio_tokens = audio_tokens.permute(0, 2, 3, 1) 
		audio_tokens = audio_tokens + self.audio_freq_pos_embed + self.audio_time_pos_embed
		
		audio_tokens = audio_tokens.flatten(1, 2) 

		audio_type_embedding = self.modality_type_embeddings(
			torch.ones(1, 1, device=audio_x.device).long()
		)
		audio_tokens = audio_tokens + audio_type_embedding

		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		
		x = torch.cat((cls_tokens, video_tokens, audio_tokens), dim=1)
		x = self.layernorm(x)
		x = self.dropout(x)
		return x
