import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput

import librosa
import re



from typing import Optional, Tuple
_CONFIG_FOR_DOC = "Wav2Vec2Config"





class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    # @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        frame_num=None

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
            Labels for connectionist temporal classification. Note that ``target_length`` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in ``[-100, 0, ..., config.vocab_size -
            1]``. All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ...,
            config.vocab_size - 1]``.

        Returns:

        Example::

            >>> import torch
            >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = torch.argmax(logits, dim=-1)

            >>> transcription = processor.decode(predicted_ids[0])

            >>> # compute loss
            >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

            >>> # wrap processor as target processor to encode labels
            >>> with processor.as_target_processor():
            >>>     labels = processor(transcription, return_tensors="pt").input_ids

            >>> loss = model(input_values, labels=labels).loss
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        hidden_states = outputs[0]

        if hidden_states.shape[1] % 2 != 0:
            # hidden_states = hidden_states[:, :-1]
            pad = (0, 0, 0, 1)
            hidden_states = F.pad(hidden_states, pad)


        if frame_num and hidden_states.shape[1] > frame_num * 2:
            hidden_states = hidden_states[:, :frame_num * 2]

        # hidden_states = self.dropout(hidden_states)
        return  hidden_states






