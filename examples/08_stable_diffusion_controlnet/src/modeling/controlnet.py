'''
 FILENAME:      controlnet.py

 AUTHORS:       Pan Shaohua

 START DATE:    Tuesday March 14th 2023

 CONTACT:       shaohua.pan@quvideo.com
'''


from typing import Optional, Tuple, Union

# import torch
# from torch import nn
# from torch.nn import functional as F

from aitemplate.frontend import nn
from aitemplate.compiler import ops

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import UNetMidBlock2DCrossAttn, get_down_block


class SiLU(nn.Module):
    def __init__(self) -> None:
        super(SiLU, self).__init__()
        self.silu = ops.silu
    
    def forward(self, x):
        out = self.silu(x)
        return out


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        # self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        # self.conv_in = nn.Conv2dBias(conditioning_channels, block_out_channels[0], 3, 1, 1)
        self.conv_in = nn.Conv2dBiasFewChannels(conditioning_channels, block_out_channels[0], 3, 1, 1, auto_padding=False)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.Conv2dBias(channel_in, channel_in, 3, 1, 1))
            self.blocks.append(nn.Conv2dBias(channel_in, channel_out, 3, 2, 1))

        # self.conv_out = zero_module(
        #     # nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        #     nn.Conv2dBias(block_out_channels[-1], conditioning_embedding_channels, 3, 1, 1)
        # )
        self.conv_out = nn.Conv2dBias(block_out_channels[-1], conditioning_embedding_channels, 3, 1, 1)
        
        self.silu = SiLU()

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        # embedding = F.silu(embedding)
        embedding = self.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            # embedding = F.silu(embedding)
            embedding = self.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        ):
        super().__init__()

        # input
        # conv_in_kernel = 3
        # conv_in_padding = (conv_in_kernel - 1) // 2
        # self.conv_in = nn.Conv2d(
        #     in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        # )
        self.conv_in = nn.Conv2dBias(in_channels, block_out_channels[0], 3, 1, 1)

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim
        )
        
        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        # controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1, 1, 0)
        # controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                # controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1, 1, 0)
                # controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                # controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1, 1, 0)
                # controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        # controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = nn.Conv2dBias(mid_block_channel, mid_block_channel, 1, 1, 0)
        # controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
        )

    def forward(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ):

        # 1. time
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        down_block_res_samples.append(mid_block_res_sample)

        return down_block_res_samples


# def zero_module(module):
#     for p in module.parameters():
#         nn.init.zeros_(p)
#     return module