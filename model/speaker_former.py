import torch
import torch.nn as nn
from model.wav2vec2focctc import Wav2Vec2ForCTC
from model.utils import init_biased_mask, enc_dec_mask, PeriodicPositionalEncoding

class SpeakFormer(nn.Module):
    def __init__(self, img_size=224, feature_dim = 256, period = 25, max_seq_len = 751,  device = 'cpu'):
        super(SpeakFormer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vector: (batch_size, seq_len, V*3)
        """

        self.img_size = img_size
        self.feature_dim = feature_dim

        # wav2vec 2.0 weights initialization

        self.audio_encoder = Wav2Vec2ForCTC.from_pretrained("external/facebook/wav2vec2-base-960h")
        self.audio_encoder.freeze_feature_extractor()
        self.audio_feature_map = nn.Linear(768, feature_dim)

        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)


        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2*feature_dim, batch_first=True)
        self.speaker_transformer_decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.device = device


    def forward(self, video_features, audio):
        # def forward(self, audio, vertice, one_hot, criterion,teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        # obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        # video: (B,T,C,H,W)
        # audio: (B,A)
        frame_num = video_features.shape[1]
        hidden_states = self.audio_encoder(audio, frame_num=frame_num)

        if hidden_states.shape[1]<frame_num*2:
            video_features = video_features[:, : hidden_states.shape[1]//2]
            frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states)
        memory_mask = enc_dec_mask(self.device, video_features.shape[1], hidden_states.shape[1])

        video_features = self.PPE(video_features)
        tgt_mask = self.biased_mask[:, :video_features.shape[1], :video_features.shape[1]].clone().detach().to(device=self.device).repeat(video_features.shape[0],1,1)
        speaker_vector = self.speaker_transformer_decoder1(video_features, video_features, tgt_mask=tgt_mask)

        speaker_vector = self.speaker_transformer_decoder2(speaker_vector, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        speaker_motion = self.speaker_transformer_decoder3(speaker_vector, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return  speaker_motion, hidden_states, speaker_vector




