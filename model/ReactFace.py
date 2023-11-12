# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import copy
# import math
# import argparse
# import librosa
# from .VAE import VanillaVAE
# from .speaker_former import SpeakFormer
# from .utils import  init_biased_mask, init_biased_mask2, enc_dec_mask, enc_dec_mask2, enc_dec_mask3, PeriodicPositionalEncoding, PositionalEncoding, get_tgt_mask
# from .video_encoder import VideoEncoder
#
#
# class Decoder(nn.Module):
#
#     def __init__(self, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 256, period = 8, max_seq_len = 751, device = 'cpu', window_size =16):
#         super(Decoder, self).__init__()
#
#         self.feature_dim = feature_dim
#
#         decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2*feature_dim, batch_first=True)
#         self.PE = PositionalEncoding(feature_dim)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
#
#         self.transformer_fusion_decode1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
#         self.transformer_fusion_decode2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
#         self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
#         self.listener_reaction_emotion_map_layer = nn.Sequential(
#             nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
#             nn.Linear(feature_dim, output_emotion_dim)
#         )
#
#
#
#         # temporal bias
#         self.biased_mask = init_biased_mask(n_head = 8, max_seq_len = max_seq_len, period=period)
#         self.period = period
#         # motion decoder
#         self.device = device
#         self.window_size = window_size
#
#
#     def forward(self, motion_sample, speaker_motion, speaker_audio):
#         B, T, _ = speaker_motion.shape
#
#         speaker_motion = speaker_motion[:,T-self.window_size:]
#         speaker_audio = speaker_audio[:, (T - self.window_size)*2: ]
#         time_queries = torch.zeros(B, T, self.feature_dim, device=speaker_motion.get_device())
#         time_queries = self.PE(time_queries)[:,T-self.window_size:]
#
#         # Pass through the transformer decoder
#         # with the latent vector for memory
#         listener_reaction = self.transformer_decoder(tgt=time_queries, memory=motion_sample)
#
#
#         tgt_mask = self.biased_mask[:, :self.window_size, :self.window_size].clone().detach().to(device=self.device).repeat(B,1,1)
#         memory_mask  = init_biased_mask2(n_head=8, window_size=self.window_size, max_seq_len=self.window_size*2, period=self.period).clone().detach().to(device=self.device).repeat(B,1,1)
#
#         listener_reaction = self.transformer_fusion_decode1(tgt=listener_reaction, memory=speaker_audio, tgt_mask=tgt_mask, memory_mask=memory_mask)
#
#         memory_mask  = init_biased_mask2(n_head=8, window_size=self.window_size, max_seq_len=self.window_size, period=self.period).clone().detach().to(device=self.device).repeat(B,1,1)
#
#         listener_reaction = self.transformer_fusion_decode2(tgt=listener_reaction, memory=speaker_motion,
#                                                          tgt_mask=tgt_mask, memory_mask=memory_mask)
#
#         listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)
#         listener_emotion_out = self.listener_reaction_emotion_map_layer(
#             torch.cat((listener_3dmm_out, listener_reaction), dim=-1))
#         return listener_3dmm_out, listener_emotion_out
#
#
#     def reset_window_size(self, window_size):
#         self.window_size = window_size
#
#
#
#
# class ReactFace(nn.Module):
#     def __init__(self, img_size=224, output_3dmm_dim = 58 , output_emotion_dim = 25, feature_dim = 256, period = 8, max_seq_len = 751, device = 'cpu', window_size = 16, momentum = 0.9):
#         super(ReactFace, self).__init__()
#         """
#         audio: (batch_size, raw_wav)
#         template: (batch_size, V*3)
#         vector: (batch_size, seq_len, V*3)
#         """
#
#         self.img_size = img_size
#         self.feature_dim = feature_dim
#         self.output_3dmm_dim = output_3dmm_dim
#         self.output_emotion_dim = output_emotion_dim
#         self.window_size = window_size
#         self.momentum = momentum
#         self.device = device
#         self.period = period
#
#
#
#         # periodic positional encoding
#         self.PPE = PeriodicPositionalEncoding(feature_dim, period = period, max_seq_len = max_seq_len)
#         self.PE = PositionalEncoding(feature_dim)
#
#         self.past_motion_linear = nn.Linear(output_3dmm_dim, feature_dim)
#
#
#         # temporal bias
#         self.biased_mask = init_biased_mask(n_head = 8, max_seq_len = max_seq_len, period=period)
#         # speaker_former
#         decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2*feature_dim, batch_first=True)
#         self.speaker_reaction_decoder = SpeakFormer(img_size=img_size, feature_dim = feature_dim, period =period, max_seq_len = max_seq_len,  device = device)
#
#         self.speaker_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
#         self.speaker_vector_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
#
#         self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
#         nn.init.constant_(self.speaker_vector_map_layer.weight, 0)
#         nn.init.constant_(self.speaker_vector_map_layer.bias, 0)
#
#         # motion decoder
#
#         self.transformer_fusion_plm2sm = nn.TransformerDecoder(decoder_layer, num_layers=1)
#         self.transformer_fusion_plm2sa = nn.TransformerDecoder(decoder_layer, num_layers=1)
#
#         self.interaction_VAE  = VanillaVAE(feature_dim, latent_dim = feature_dim)
#
#
#         self.transformer_listener_vectors = nn.TransformerDecoder(decoder_layer, num_layers=2)
#
#
#         self.listener_reaction_decoder = Decoder(output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim, period = period, max_seq_len = max_seq_len, device=device, window_size = window_size)
#
#
#
#     def speaker_motion_past_listener_motion_to_motion(self, speaker_motion, speaker_audio, past_listener_motions, listener_vectors_=None):
#
#         frame_num = past_listener_motions.shape[1]
#         B = speaker_motion.shape[0]
#
#         speaker_audio = speaker_audio[:,:2*frame_num]
#         speaker_motion = speaker_motion[:,:frame_num]
#
#
#         tgt_mask = self.biased_mask[:, :frame_num, :frame_num].clone().detach().to(device=self.device).repeat(B, 1, 1)
#
#
#         memory_mask  = init_biased_mask2(n_head=8, window_size=frame_num, max_seq_len=speaker_audio.shape[1], period=self.period).clone().detach().to(device=self.device).repeat(B,1,1)
#         listener_motion = self.transformer_fusion_plm2sa(tgt=past_listener_motions, memory=speaker_audio,
#                                                          tgt_mask=tgt_mask, memory_mask=memory_mask)
#
#         memory_mask  = init_biased_mask2(n_head=8, window_size=frame_num, max_seq_len=speaker_motion.shape[1], period=self.period).clone().detach().to(device=self.device).repeat(B,1,1)
#         listener_motion = self.transformer_fusion_plm2sm(tgt=listener_motion, memory=speaker_motion,
#                                                          tgt_mask=tgt_mask, memory_mask=memory_mask)
#
#         tgt_mask = self.biased_mask[:, :self.window_size, :self.window_size].clone().detach().to(device=self.device).repeat(B, 1, 1)
#         if listener_vectors_ is not None:
#             listener_cur_motion = self.transformer_listener_vectors(tgt=listener_vectors_, memory=listener_vectors_,
#                                                                     tgt_mask=tgt_mask)
#
#             listener_motion = torch.cat((listener_motion, listener_cur_motion), dim = 1)
#         motion_sample, distribution = self.interaction_VAE(listener_motion)
#
#         return  motion_sample, distribution
#
#
#     def forward(self, speaker_videos, speaker_audios, speaker_out =  False):
#         encoded_speaker_features = self.video_encoder(speaker_videos)
#         speaker_motion, speaker_audio, speaker_vector = self.speaker_reaction_decoder(encoded_speaker_features, speaker_audios)
#         frame_num = speaker_motion.shape[1]
#         B = speaker_motion.shape[0]
#
#         past_reaction_3dmm = torch.zeros((speaker_videos.size(0), self.window_size, self.output_3dmm_dim),
#                                     device=speaker_videos.get_device())
#         past_reaction_emotion = torch.zeros((speaker_videos.size(0), self.window_size, self.output_emotion_dim),
#                                        device=speaker_videos.get_device())
#         distribution = []
#         past_motion_sample = None
#         for i in range(0, frame_num // self.window_size):
#
#             speaker_motion_, speaker_audio_ = speaker_motion[:, : (i + 1) * self.window_size], speaker_audio[:, : 2 * (
#                         i + 1) * self.window_size]
#
#
#             pre_listener_motion_ = self.past_motion_linear(past_reaction_3dmm)
#             pre_listener_motion_ += self.PPE(pre_listener_motion_)
#
#             motion_sample, dis = self.speaker_motion_past_listener_motion_to_motion(speaker_motion_,
#                                                                                                speaker_audio_,
#                                                                                                pre_listener_motion_)
#
#             distribution.append(dis)
#             if past_motion_sample is not None:
#                 motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
#                 motion_sample_input = F.interpolate(
#                     torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
#                     self.window_size, mode='linear')
#                 motion_sample_input = motion_sample_input.transpose(1, 2)
#             else:
#                 motion_sample_input = motion_sample.unsqueeze(1)
#
#             past_motion_sample = motion_sample
#
#             listener_3dmm_out, listener_emotion_out = self.listener_reaction_decoder(motion_sample_input, speaker_motion_, speaker_audio_)
#
#             if i != 0:
#                 past_reaction_3dmm = torch.cat((past_reaction_3dmm, listener_3dmm_out), 1)
#                 past_reaction_emotion = torch.cat((past_reaction_emotion, listener_emotion_out), 1)
#             else:
#                 past_reaction_3dmm = listener_3dmm_out
#                 past_reaction_emotion = listener_emotion_out
#
#         if speaker_out:
#             speaker_vector = self.speaker_decoder(speaker_vector, speaker_vector)
#             speaker_3dmm_out = self.speaker_vector_map_layer(speaker_vector)
#             return past_reaction_3dmm, past_reaction_emotion, distribution, speaker_3dmm_out
#
#         return  past_reaction_3dmm, past_reaction_emotion, distribution
#
#
#
#     def reset_window_size(self, window_size):
#         self.window_size = window_size
#         self.listener_reaction_decoder.reset_window_size(window_size)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import argparse
import librosa
from .VAE import VanillaVAE
from .speaker_former import SpeakFormer
from .utils import init_biased_mask, init_biased_mask2, enc_dec_mask, enc_dec_mask2, enc_dec_mask3, \
    PeriodicPositionalEncoding, PositionalEncoding, get_tgt_mask
from .video_encoder import VideoEncoder


class Decoder(nn.Module):

    def __init__(self, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=256, period=8, max_seq_len=751,
                 device='cpu', window_size=16):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2 * feature_dim,
                                                   batch_first=True)
        self.PE = PositionalEncoding(feature_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.transformer_fusion_decode1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.transformer_fusion_decode2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )

        # temporal bias
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)
        self.period = period
        # motion decoder
        self.device = device
        self.window_size = window_size

    def forward(self, motion_sample, speaker_motion, speaker_audio, past_reaction_emotion, past_reaction_3dmm):
        B, T, _ = speaker_motion.shape

        speaker_motion = speaker_motion[:, T - self.window_size:]
        speaker_audio = speaker_audio[:, (T - self.window_size) * 2:]
        time_queries = torch.zeros(B, T, self.feature_dim, device=speaker_motion.get_device())
        time_queries = self.PE(time_queries)[:, T - self.window_size:]

        # Pass through the transformer decoder
        # with the latent vector for memory
        listener_reaction = self.transformer_decoder(tgt=time_queries, memory=motion_sample)

        tgt_mask = self.biased_mask[:, :self.window_size, :self.window_size].clone().detach().to(
            device=self.device).repeat(B, 1, 1)
        memory_mask = init_biased_mask2(n_head=8, window_size=self.window_size, max_seq_len=self.window_size * 2,
                                        period=self.period).clone().detach().to(device=self.device).repeat(B, 1, 1)

        listener_reaction = self.transformer_fusion_decode1(tgt=listener_reaction, memory=speaker_audio,
                                                            tgt_mask=tgt_mask, memory_mask=memory_mask)

        memory_mask = init_biased_mask2(n_head=8, window_size=self.window_size, max_seq_len=self.window_size,
                                        period=self.period).clone().detach().to(device=self.device).repeat(B, 1, 1)

        listener_reaction = self.transformer_fusion_decode2(tgt=listener_reaction, memory=speaker_motion,
                                                            tgt_mask=tgt_mask, memory_mask=memory_mask)

        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)
        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        # listener_3dmm_out, listener_emotion_out =  listener_3dmm_out + past_reaction_3dmm, listener_emotion_out + past_reaction_emotion

        return listener_3dmm_out, listener_emotion_out

    def reset_window_size(self, window_size):
        self.window_size = window_size


class ReactFace(nn.Module):
    def __init__(self, img_size=224, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=256, period=8,
                 max_seq_len=751, device='cpu', window_size=16, momentum=0.9):
        super(ReactFace, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vector: (batch_size, seq_len, V*3)
        """

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.window_size = window_size
        self.momentum = momentum
        self.device = device
        self.period = period

        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.PE = PositionalEncoding(feature_dim)

        self.past_motion_linear = nn.Linear(output_3dmm_dim, feature_dim)

        # temporal bias
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)
        # speaker_former
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2 * feature_dim,
                                                   batch_first=True)
        self.speaker_reaction_decoder = SpeakFormer(img_size=img_size, feature_dim=feature_dim, period=period,
                                                    max_seq_len=max_seq_len, device=device)

        self.speaker_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_vector_map_layer = nn.Linear(feature_dim, output_3dmm_dim)

        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        nn.init.constant_(self.speaker_vector_map_layer.weight, 0)
        nn.init.constant_(self.speaker_vector_map_layer.bias, 0)

        # motion decoder

        self.transformer_fusion_plm2sm = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.transformer_fusion_plm2sa = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.interaction_VAE = VanillaVAE(feature_dim, latent_dim=feature_dim)

        self.transformer_listener_vectors = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.listener_reaction_decoder = Decoder(output_3dmm_dim=output_3dmm_dim, output_emotion_dim=output_emotion_dim,
                                                 feature_dim=feature_dim, period=period, max_seq_len=max_seq_len,
                                                 device=device, window_size=window_size)

    def speaker_motion_past_listener_motion_to_motion(self, speaker_motion, speaker_audio, past_listener_motions,
                                                      listener_vectors_=None):

        frame_num = past_listener_motions.shape[1]
        B = speaker_motion.shape[0]

        speaker_audio = speaker_audio[:, :2 * frame_num]
        speaker_motion = speaker_motion[:, :frame_num]

        tgt_mask = self.biased_mask[:, :frame_num, :frame_num].clone().detach().to(device=self.device).repeat(B, 1, 1)

        memory_mask = init_biased_mask2(n_head=8, window_size=frame_num, max_seq_len=speaker_audio.shape[1],
                                        period=self.period).clone().detach().to(device=self.device).repeat(B, 1, 1)
        listener_motion = self.transformer_fusion_plm2sa(tgt=past_listener_motions, memory=speaker_audio,
                                                         tgt_mask=tgt_mask, memory_mask=memory_mask)

        memory_mask = init_biased_mask2(n_head=8, window_size=frame_num, max_seq_len=speaker_motion.shape[1],
                                        period=self.period).clone().detach().to(device=self.device).repeat(B, 1, 1)
        listener_motion = self.transformer_fusion_plm2sm(tgt=listener_motion, memory=speaker_motion,
                                                         tgt_mask=tgt_mask, memory_mask=memory_mask)

        tgt_mask = self.biased_mask[:, :self.window_size, :self.window_size].clone().detach().to(
            device=self.device).repeat(B, 1, 1)
        if listener_vectors_ is not None:
            listener_cur_motion = self.transformer_listener_vectors(tgt=listener_vectors_, memory=listener_vectors_,
                                                                    tgt_mask=tgt_mask)

            listener_motion = torch.cat((listener_motion, listener_cur_motion), dim=1)
        motion_sample, distribution = self.interaction_VAE(listener_motion)

        return motion_sample, distribution

    def forward(self, speaker_videos, speaker_audios, speaker_out=False):
        encoded_speaker_features = self.video_encoder(speaker_videos)
        speaker_motion, speaker_audio, speaker_vector = self.speaker_reaction_decoder(encoded_speaker_features,
                                                                                      speaker_audios)
        frame_num = speaker_motion.shape[1]
        B = speaker_motion.shape[0]

        past_reaction_3dmm = torch.zeros((speaker_videos.size(0), self.window_size, self.output_3dmm_dim),
                                         device=speaker_videos.get_device())
        past_reaction_emotion = torch.zeros((speaker_videos.size(0), self.window_size, self.output_emotion_dim),
                                            device=speaker_videos.get_device())
        distribution = []
        past_motion_sample = None
        for i in range(0, frame_num // self.window_size):

            speaker_motion_, speaker_audio_ = speaker_motion[:, : (i + 1) * self.window_size], speaker_audio[:, : 2 * (
                    i + 1) * self.window_size]

            pre_listener_motion_ = self.past_motion_linear(past_reaction_3dmm)
            pre_listener_motion_ += self.PPE(pre_listener_motion_)

            motion_sample, dis = self.speaker_motion_past_listener_motion_to_motion(speaker_motion_,
                                                                                    speaker_audio_,
                                                                                    pre_listener_motion_)

            distribution.append(dis)
            if past_motion_sample is not None:
                motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
                motion_sample_input = F.interpolate(
                    torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
                    self.window_size, mode='linear')
                motion_sample_input = motion_sample_input.transpose(1, 2)
            else:
                motion_sample_input = motion_sample.unsqueeze(1)

            past_motion_sample = motion_sample

            listener_3dmm_out, listener_emotion_out = self.listener_reaction_decoder(motion_sample_input, speaker_motion_, speaker_audio_, past_reaction_emotion[:, -1].unsqueeze(1), past_reaction_3dmm[:, -1].unsqueeze(1))

            if i != 0:
                past_reaction_3dmm = torch.cat((past_reaction_3dmm, listener_3dmm_out), 1)
                past_reaction_emotion = torch.cat((past_reaction_emotion, listener_emotion_out), 1)
            else:
                past_reaction_3dmm = listener_3dmm_out
                past_reaction_emotion = listener_emotion_out

        if speaker_out:
            speaker_vector = self.speaker_decoder(speaker_vector, speaker_vector)
            speaker_3dmm_out = self.speaker_vector_map_layer(speaker_vector)
            return past_reaction_3dmm, past_reaction_emotion, distribution, speaker_3dmm_out

        return past_reaction_3dmm, past_reaction_emotion, distribution

    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.listener_reaction_decoder.reset_window_size(window_size)
