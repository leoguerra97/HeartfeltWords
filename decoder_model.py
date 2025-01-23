import pandas as pd
pd.options.mode.chained_assignment = None

import torch
import torch.nn as nn

from torch.jit.annotations import Optional

from transformers import GPT2LMHeadModel, T5Tokenizer, T5Model
from typing import Tuple


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())

        self.model = nn.Sequential(*layers)

class CaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.encoded_ecg_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, pretrain=False):
        embedding_text = self.gpt.transformer.wte(tokens)

        if pretrain:
            out = self.gpt(inputs_embeds=embedding_text, labels=labels, attention_mask=mask)
            return out
        prefix_projections = self.ecg_project(prefix).view(-1, self.encoded_ecg_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # embedding_cat = self.pos_en(embedding_cat) #POSITIONAL ENCODING Layer

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, tokenizer, encoded_ecg_length: int, encoded_ecg_size: int):
        super(CaptionModel, self).__init__()
        self.device = torch.device('cuda:0')
        self.encoded_ecg_length = encoded_ecg_length
        self.encoded_ecg_size = encoded_ecg_size
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        #self.gpt = T5Tokenizer.from_pretrained("t5-base")

        self.gpt.resize_token_embeddings(len(tokenizer))
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.ecg_project = MLP((encoded_ecg_size, self.gpt_embedding_size // 2,
                                self.gpt_embedding_size))

        # ecg_project = self.ecg_project.to(self.device)
        # summary(ecg_project, (encoded_ecg_length,encoded_ecg_size))

class FullModel(nn.Module):
    def __init__(self, caption_model, encoder_model, tokenizer):
        super(FullModel, self).__init__()
        self.device = torch.device('cuda:0')
        self.encoder = encoder_model
        self.decoder = caption_model
        self.tokenizer = tokenizer

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, pretrain=False):
        classif, prefix = self.encoder(prefix)
        prefix = prefix.view(-1, self.decoder.encoded_ecg_length, self.decoder.encoded_ecg_size)
        x = self.decoder(tokens, prefix, mask)
        return x

