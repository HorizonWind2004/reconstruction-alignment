import torch
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX
from .harmon import Harmon


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class HarmonDev(Harmon, BaseModel):
    def __init__(self,
                 grad_scale=0.1,
                 loss_weights={'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0},
                 pretrained_pth=None,
                 freeze_llm=False,
                 gradient_checkpointing=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.debuged = False

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        if freeze_llm:
            self.llm.requires_grad_(False)

        # gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k}

        return state_dict

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        x_enc = self.forward_mae_encoder(x, mask, input_ids=input_ids,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def image2text_loss(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            assert False
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            for i in range(input_ids.shape[0]):
                assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
            input_ids = torch.cat(
                (
                    input_ids[:, :3],
                    IMAGE_TOKEN_INDEX * torch.ones(
                        input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                    input_ids[:, 4:],
                ), dim=1
            )
            attention_mask = torch.cat(
                (
                    attention_mask[:, :3],
                    torch.ones(
                        attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                    attention_mask[:, 4:],
                ), dim=1
            )
            labels = torch.cat(
                (
                    labels[:, :3],
                    labels[0, 3] * torch.ones(
                        labels.shape[0], 1088 , dtype=torch.long, device=labels.device),
                    labels[:, 4:],
                ), dim=1
            )
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            loss_null = 0.0

        output = self.llm_model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                return_dict=True)

        last_hidden_state = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        last_hidden_state = last_hidden_state[labels >= 0]
        labels = labels[labels >= 0]
        logits = self.llm.get_output_embeddings()(last_hidden_state)

        loss_i2t = F.cross_entropy(input=logits, target=labels)

        return loss_i2t + loss_null

    def recon_loss(self, data_dict):

        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        _, z_enc = self.extract_visual_feature(x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
            elif 'image2text' in data_type:
                loss = self.image2text_loss(batch_data)
            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
            else:
                raise NotImplementedError(f"Unknown data type: {data_type}")
            losses[f'loss_{data_type}'] = loss * self.loss_weights[data_type]
        return losses
