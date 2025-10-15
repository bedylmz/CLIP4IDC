import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from logging import Logger

from modules.siglip.modeling_siglip_txt import SiglipTextModel, SiglipTextConfig
from modules.siglip.modeling_siglip_view import SiglipVisionModel, SiglipVisionConfig


class Siglip(nn.Module):
    def __init__(
        self,
        text_config: SiglipTextConfig = None,
        vision_config: SiglipVisionConfig = None,
    ):
        super().__init__()
        if text_config is None:
            text_config = SiglipTextConfig()

        if vision_config is None:
            vision_config = SiglipVisionConfig()

        self.vision_config = vision_config
        self.text_config = text_config

        self.visual_model = SiglipVisionModel(self.vision_config)
        self.text_model = SiglipTextModel(self.text_config)

        self.logit_scale = nn.Parameter(torch.ones([]))

    @staticmethod
    def get_config(self):

        config = {}
        config["text_config"] = str(self.text_config)
        config["visual_config"] = str(self.visual_config)

        return config

    def encode_images(self, image_pair, return_hidden=False):
        image_hidden = self.visual_model(image_pair)

        # [16, 196, 768]
        # print(" image_hidden* " * 10)
        # print(image_hidden.shape)
        # print(" image_hidden* " * 10)

        image_hidden = image_hidden.view(
            int(image_hidden.shape[0] / 2), int(image_hidden.shape[1] * 2), int(image_hidden.shape[2])
        )

        # [8, 392, 768]
        # print(" image_hidden reshaped* " * 10)
        # print(image_hidden.shape)
        # print(" image_hidden reshaped* " * 10)

        X = torch.cat(
            [image_hidden[:, 0, :].unsqueeze(1), image_hidden[:, 196, :].unsqueeze(1)],
            1,
        )

        # [16, 2, 768] , reshaped = torch.Size([8, 2, 768])
        # print(" Visual X* " * 10)
        # print(X.shape)
        # print(" Visual X* " * 10)

        X = torch.mean(X, 1)

        # [16, 768] , reshaped = torch.Size([8, 768])
        # print(" Visual X* " * 10)
        # print(X.shape)
        # print(" Visual X* " * 10)

        if return_hidden:
            return X, image_hidden

        return X

    def encode_text(self, text: str, return_hidden=False):
        hidden = self.text_model(text)

        # [8, 64, 768]
        # print(" Text hidden* " * 10)
        # print(hidden.shape)
        # print(" Text hidden* " * 10)

        X = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        # [8, 768]
        # print(" Text X* " * 10)
        # print(X.shape)
        # print(" Text X* " * 10)

        if return_hidden:
            return X, hidden

        return X

    def forward(self, image, text):
        image_features = self.encode_images(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features * text_features
        logits_per_text = logit_scale * text_features * image_features

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
