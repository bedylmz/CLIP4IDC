import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from safetensors.torch import load_file

from modules.module_decoder import DecoderModel, DecoderConfig
from modules.until_module import PreTrainedModel, AllGather, CrossEn

from modules.module_clip import CLIP, convert_weights

from modules.siglip.module_siglip import Siglip


# class Siglip4IDCPreTrainedModel(nn.Module):
#     def __init__(
#         self,
#     ):
#         super().__init__()

#     @classmethod
#     def from_pretrained(
#         cls,
#         pretrained_model_path: str,
#     ):
#         if pretrained_model_path is None:
#             raise ValueError("pretrained_model_path cannot be None")

#         # Load the model
#         model = cls()

#         state_dict = torch.load(pretrained_model_path, map_location="cpu")
#         # Convert the state dict to the model's format
#         vision_siglip_state_dict = {k: v for k, v in state_dict.items() if k.startswith("vision_model.")}
#         text_siglip_state_dict = {k: v for k, v in state_dict.items() if k.startswith("text_model.")}
#         # Load the weights

#         model.visual_model.load_state_dict(vision_siglip_state_dict, strict=False)
#         model.text_model.load_state_dict(text_siglip_state_dict, strict=False)

#         return model


class Siglip4IDC(nn.Module):
    def __init__(self, pretrained_model_path=None):
        super(Siglip4IDC, self).__init__()

        self.siglip = Siglip()
        self.loss_fct = CrossEn()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
    ):
        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path cannot be None")

        # Load the model
        model = cls()

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        # Convert the state dict to the model's format
        vision_siglip_state_dict = {k: v for k, v in state_dict.items() if k.startswith("vision_model.")}
        text_siglip_state_dict = {k: v for k, v in state_dict.items() if k.startswith("text_model.")}
        # Load the weights

        model.siglip.visual_model.load_state_dict(vision_siglip_state_dict, strict=False)
        model.siglip.text_model.load_state_dict(text_siglip_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        before_image,
        after_image,
        image_mask,
    ):

        #! given images should be preprocessed, tensor

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        image_mask = image_mask.view(-1, image_mask.shape[-1])

        # print("Before " * 10)
        # print(before_image.shape)
        # print("Before " * 10)

        # print("After " * 10)
        # print(after_image.shape)
        # print("After " * 10)

        #! concatenate image pairs
        image_pairs = torch.cat([before_image, after_image], dim=1)
        # print("-" * 10)
        # print(image_pairs.shape)
        # print("-" * 10)

        batch, pair, channel, height, width = image_pairs.shape
        image_pairs = image_pairs.view(batch * pair, channel, height, width)

        # print("-" * 10)
        # print(image_pairs.shape)
        # print("-" * 10)

        sequence_emb, visual_emb, sequence_output, visual_output = self.get_sequence_and_visual_output(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_pair=image_pairs,
            visual_mask=image_mask,
            shaped=True,
        )

        loss = 0.0
        sim_matrix, *_tmp = self.get_similarity_logits(
            sequence_emb,
            visual_emb,
            attention_mask,
            image_mask,
            shaped=True,
        )

        # print("Sim matrix dimension")
        # print(sim_matrix.shape)
        # print("Sim matrix dimension")

        sim_loss1 = self.loss_fct(sim_matrix)
        sim_loss2 = self.loss_fct(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2

        loss += sim_loss

        return loss

    def get_sequence_output(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        shaped=False,
    ):
        if not shaped:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_output, sequence_hidden = self.siglip.encode_text(input_ids, return_hidden=True)
        sequence_output = sequence_output.float()
        sequence_hidden = sequence_hidden.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_output, sequence_hidden

    def get_visual_output(
        self,
        image_pair,
        visual_mask,
        shaped=False,
    ):
        if shaped is False:
            visual_mask = visual_mask.view(-1, visual_mask.shape[-1])
            image_pair = torch.as_tensor(image_pair).float()
            batch, pair, channel, height, width = image_pair.shape
            image_pair = image_pair.view(batch * pair, channel, width, height)

        bs_pair = visual_mask.size(0)
        visual_output, visual_hidden = self.siglip.encode_images(image_pair, return_hidden=True)

        visual_hidden = visual_hidden.float()
        visual_output = visual_output.float()

        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_output, visual_hidden

    def get_sequence_and_visual_output(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        image_pair,
        visual_mask,
        shaped=False,
    ):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, visual_mask.shape[-1])
            visual_mask = visual_mask.view(-1, visual_mask.shape[-1])

            visual_mask = visual_mask.view(-1, visual_mask.shape[-1])
            image_pair = torch.as_tensor(image_pair).float()
            b, pair, channel, h, w = image_pair.shape
            image_pair = image_pair.view(b * pair, channel, h, w)

        sequence_output, sequence_hidden = self.get_sequence_output(
            input_ids,
            token_type_ids,
            attention_mask,
            shaped=True,
        )

        visual_output, visual_hidden = self.get_visual_output(
            image_pair,
            visual_mask,
            shaped=True,
        )

        return sequence_output, visual_output, sequence_hidden, visual_hidden

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, visual_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        visual_output = visual_output.squeeze(1)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.siglip.logit_scale.exp()

        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())

        return retrieve_logits

    def get_similarity_logits(
        self,
        sequence_output,
        visual_output,
        attention_mask,
        visual_mask,
        shaped=False,
    ):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            visual_mask = visual_mask.view(-1, visual_mask.shape[-1])

        contrastive_direction = ()

        retrieve_logits = self._loose_similarity(
            sequence_output,
            visual_output,
            attention_mask,
            visual_mask,
        )

        return retrieve_logits, contrastive_direction
