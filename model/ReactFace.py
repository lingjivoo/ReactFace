import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import Tuple, List, Optional

from model.VAE import VanillaVAE
from model.speaker_former import SpeakFormer
from model.utils import (
    init_biased_mask,
    init_biased_mask2,
    PeriodicPositionalEncoding,
    PositionalEncoding
)
from model.video_encoder import VideoEncoder


class Decoder(nn.Module):
    """
    Decoder module for generating listener reactions based on speaker motion and audio.

    Args:
        output_3dmm_dim (int): Dimension of 3DMM output features
        feature_dim (int): Dimension of internal feature representations
        period (int): Period for positional encoding
        max_seq_len (int): Maximum sequence length
        device (str): Device to run the model on ('cpu' or 'cuda')
        window_size (int): Size of the sliding window for processing
    """

    def __init__(
            self,
            output_3dmm_dim: int = 58,
            feature_dim: int = 256,
            period: int = 8,
            max_seq_len: int = 751,
            device: str = 'cpu',
            window_size: int = 16
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        self.period = period

        # Initialize transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )

        self.PE = PositionalEncoding(feature_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.transformer_fusion_decode1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.transformer_fusion_decode2 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Output mapping layers
        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)

        # Initialize temporal bias
        self.biased_mask = init_biased_mask(
            n_head=8,
            max_seq_len=max_seq_len,
            period=period
        )

    def forward(
            self,
            motion_sample: torch.Tensor,
            speaker_motion: torch.Tensor,
            speaker_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            motion_sample: Sampled motion features
            speaker_motion: Speaker motion features
            speaker_audio: Speaker audio features
        Returns:
            torch.Tensor: Generated listener 3DMM parameters
        """

        ## seq_len: involves both past len and current len
        batch_size, seq_len, _ = speaker_motion.shape

        # Slice current speaker motion and audio
        speaker_motion = speaker_motion[:, seq_len - self.window_size:]
        speaker_audio = speaker_audio[:, (seq_len - self.window_size) * 2:]

        # Generate global time queries
        time_queries = torch.zeros(
            batch_size,
            seq_len,
            self.feature_dim,
            device=speaker_motion.device
        )

        # Slice current time_queries for listener
        time_queries = self.PE(time_queries)[:, seq_len - self.window_size:]

        # Generate listener reaction through transformer decoder
        listener_reaction = self.transformer_decoder(
            tgt=time_queries,
            memory=motion_sample
        )

        # Prepare masks
        tgt_mask = self.biased_mask[:, :self.window_size, :self.window_size].clone().detach().to(
            device=self.device
        ).repeat(batch_size, 1, 1)

        memory_mask = init_biased_mask2(
            n_head=8,
            window_size=self.window_size,
            max_seq_len=self.window_size * 2,
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)

        # Apply fusion decoders
        listener_reaction = self.transformer_fusion_decode1(
            tgt=listener_reaction,
            memory=speaker_audio,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        memory_mask = init_biased_mask2(
            n_head=8,
            window_size=self.window_size,
            max_seq_len=self.window_size,
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)

        listener_reaction = self.transformer_fusion_decode2(
            tgt=listener_reaction,
            memory=speaker_motion,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        # Map to 3DMM parameters
        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)
        return listener_3dmm_out

    def reset_window_size(self, window_size: int) -> None:
        """Reset the window size for processing."""
        self.window_size = window_size


class ReactFace(nn.Module):
    """
    ReactFace model for generating listener reactions based on speaker audio and video.

    Args:
        img_size (int): Input image size
        output_3dmm_dim (int): Dimension of 3DMM output features
        feature_dim (int): Dimension of internal feature representations
        period (int): Period for positional encoding
        max_seq_len (int): Maximum sequence length
        device (str): Device to run the model on
        window_size (int): Size of the sliding window
        momentum (float): Momentum factor for motion sample updates
    """

    def __init__(
            self,
            img_size: int = 224,
            output_3dmm_dim: int = 58,
            feature_dim: int = 256,
            period: int = 8,
            max_seq_len: int = 751,
            device: str = 'cpu',
            window_size: int = 16,
            momentum: float = 0.9
    ):
        super().__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.window_size = window_size
        self.momentum = momentum
        self.device = device
        self.period = period

        # Initialize positional encodings
        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.PE = PositionalEncoding(feature_dim)

        # Initialize linear layers
        self.past_motion_linear = nn.Linear(output_3dmm_dim, feature_dim)

        # Initialize transformer components
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )

        # Initialize various model components
        self.speaker_reaction_decoder = SpeakFormer(
            img_size=img_size,
            feature_dim=feature_dim,
            period=period,
            max_seq_len=max_seq_len,
            device=device
        )
        self.speaker_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_vector_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)

        # Initialize weights
        nn.init.constant_(self.speaker_vector_map_layer.weight, 0)
        nn.init.constant_(self.speaker_vector_map_layer.bias, 0)

        # Initialize fusion transformers
        self.transformer_fusion_plm2sm = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.transformer_fusion_plm2sa = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Initialize VAE and listener components
        self.interaction_VAE = VanillaVAE(feature_dim, latent_dim=feature_dim)
        self.transformer_listener_vectors = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.listener_reaction_decoder = Decoder(
            output_3dmm_dim=output_3dmm_dim,
            feature_dim=feature_dim,
            period=period,
            max_seq_len=max_seq_len,
            device=device,
            window_size=window_size
        )

        # Initialize temporal bias
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)

    def speaker_motion_past_listener_motion_to_motion(
            self,
            speaker_motion: torch.Tensor,
            speaker_audio: torch.Tensor,
            past_listener_motions: torch.Tensor,
            listener_vectors_: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process speaker motion and past listener motion to generate new motion.

        Args:
            speaker_motion: Speaker motion features
            speaker_audio: Speaker audio features
            past_listener_motions: Past listener motion features
            listener_vectors_: Optional listener vector features

        Returns:
            Tuple containing:
                - Generated motion sample
                - Distribution parameters
        """
        frame_num = past_listener_motions.shape[1]
        batch_size = speaker_motion.shape[0]

        # Process audio and motion inputs
        speaker_audio = speaker_audio[:, :2 * frame_num]
        speaker_motion = speaker_motion[:, :frame_num]

        # Prepare masks
        tgt_mask = self.biased_mask[:, :frame_num, :frame_num].clone().detach().to(
            device=self.device
        ).repeat(batch_size, 1, 1)

        memory_mask = init_biased_mask2(
            n_head=8,
            window_size=frame_num,
            max_seq_len=speaker_audio.shape[1],
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)

        # Apply fusion transformers
        past_listener_motions = self.transformer_fusion_plm2sa(
            tgt=past_listener_motions,
            memory=speaker_audio,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        memory_mask = init_biased_mask2(
            n_head=8,
            window_size=frame_num,
            max_seq_len=speaker_motion.shape[1],
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)

        past_listener_motions = self.transformer_fusion_plm2sm(
            tgt=past_listener_motions,
            memory=speaker_motion,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        # Generate motion sample through VAE
        motion_sample, distribution = self.interaction_VAE(past_listener_motions)
        return motion_sample, distribution

    def forward(
            self,
            speaker_videos: torch.Tensor,
            speaker_audios: torch.Tensor,
            speaker_out: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the ReactFace model.

        Args:
            speaker_videos: Input speaker video features
            speaker_audios: Input speaker audio features
            speaker_out: Whether to output speaker features

        Returns:
            Tuple containing:
                - Generated listener 3DMM parameters
                - List of distribution parameters
                - Optional speaker 3DMM parameters
        """
        # Encode speaker features
        encoded_speaker_features = self.video_encoder(speaker_videos)
        speaker_motion, speaker_audio, speaker_vector = self.speaker_reaction_decoder(
            encoded_speaker_features,
            speaker_audios
        )

        frame_num = speaker_motion.shape[1]

        # Initialize past reaction tensors
        past_reaction_3dmm = torch.zeros(
            (speaker_videos.size(0), self.window_size, self.output_3dmm_dim),
            device=speaker_videos.device
        )
        distribution = []
        past_motion_sample = None
        iterations = math.ceil(frame_num / self.window_size)

        # Process sequence in windows
        for i in range(iterations):
            speaker_motion_ = speaker_motion[:, :(i + 1) * self.window_size]
            speaker_audio_ = speaker_audio[:, :2 * (i + 1) * self.window_size]

            # Process past motion
            pre_listener_motion_ = self.past_motion_linear(past_reaction_3dmm)
            pre_listener_motion_ += self.PPE(pre_listener_motion_)

            # Generate motion sample
            motion_sample, dis = self.speaker_motion_past_listener_motion_to_motion(
                speaker_motion_,
                speaker_audio_,
                pre_listener_motion_
            )

            distribution.append(dis)

            # Apply momentum if past motion exists
            if past_motion_sample is not None:
                motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
                motion_sample_input = F.interpolate(
                    torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
                    self.window_size,
                    mode='linear'
                )
                motion_sample_input = motion_sample_input.transpose(1, 2)
            else:
                motion_sample_input = motion_sample.unsqueeze(1)

            past_motion_sample = motion_sample

            # Generate listener reactions
            listener_3dmm_out = self.listener_reaction_decoder(
                motion_sample_input,
                speaker_motion_,
                speaker_audio_,
            )

            # Concatenate or initialize past reactions
            if i != 0:
                past_reaction_3dmm = torch.cat((past_reaction_3dmm, listener_3dmm_out), 1)
            else:
                past_reaction_3dmm = listener_3dmm_out

        # Trim to match input sequence length
        past_reaction_3dmm = past_reaction_3dmm[:, :speaker_videos.shape[1]]

        # Generate speaker output if requested
        if speaker_out:
            speaker_vector = self.speaker_decoder(speaker_vector, speaker_vector)
            speaker_3dmm_out = self.speaker_vector_map_layer(speaker_vector)
            return past_reaction_3dmm, distribution, speaker_3dmm_out

        return past_reaction_3dmm, distribution

    def reset_window_size(self, window_size: int) -> None:
        """Reset the window size for the model and decoder."""
        self.window_size = window_size
        self.listener_reaction_decoder.reset_window_size(window_size)

    @torch.no_grad()
    def inference_step(
            self,
            speaker_videos: torch.Tensor,  # Shape: (B, T, C, H, W)
            speaker_audios: torch.Tensor,  # Shape: (B, T*hop_size)
            past_reaction_3dmm: torch.Tensor,  # Shape: (B, window_size, output_3dmm_dim)
            past_motion_sample: Optional[torch.Tensor] = None  # Shape: (B, feature_dim, window_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inference for the current window using partial inputs.

        Args:
            speaker_videos: Recent speaker video frames including current window
            speaker_audios: Recent speaker audio including current window
            past_reaction_3dmm: Past predicted 3DMM coefficients
            past_motion_sample: Optional past motion sample for momentum

        Returns:
            Tuple containing:
                - Current window predictions of 3DMM coefficients
                - Updated motion sample for next iteration
        """
        # Encode speaker features
        encoded_speaker_features = self.video_encoder(speaker_videos)
        speaker_motion, speaker_audio, speaker_vector = self.speaker_reaction_decoder(
            encoded_speaker_features,
            speaker_audios
        )

        # Extract features for current window
        speaker_motion = speaker_motion[:, -self.window_size:]  # Last window of motion
        speaker_audio = speaker_audio[:,
                        -2 * self.window_size:]  # Last window of audio (2x due to different sampling rate)

        # Process past motion
        pre_listener_motion = self.past_motion_linear(past_reaction_3dmm)
        pre_listener_motion = pre_listener_motion + self.PPE(pre_listener_motion)

        # Generate motion sample
        motion_sample, _ = self.speaker_motion_past_listener_motion_to_motion(
            speaker_motion,
            speaker_audio,
            pre_listener_motion
        )

        # Apply momentum if past motion exists
        if past_motion_sample is not None:
            motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
            motion_sample_input = F.interpolate(
                torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
                self.window_size,
                mode='linear'
            )
            motion_sample_input = motion_sample_input.transpose(1, 2)
        else:
            motion_sample_input = motion_sample.unsqueeze(1)

        # Generate current window predictions
        listener_3dmm_out = self.listener_reaction_decoder(motion_sample_input, speaker_motion, speaker_audio)

        return listener_3dmm_out, motion_sample