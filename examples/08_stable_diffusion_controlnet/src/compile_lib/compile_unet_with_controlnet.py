#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from ..modeling.unet_2d_condition_with_controlnet import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import mark_output


def map_unet_params(pt_mod, dim):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def compile_unet(
    pt_mod,
    batch_size=2,
    height=64,
    width=64,
    dim=320,
    hidden_dim=1024,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
    ait_so_path='./tmp'
):

    ait_mod = ait_UNet2DConditionModel(
        sample_size=64,
        cross_attention_dim=hidden_dim,
        attention_head_dim=attention_head_dim,
    )
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pt_mod.eval()
    params_ait = map_unet_params(pt_mod, dim)

    latent_model_input_ait = Tensor(
        [batch_size, height, width, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, 64, hidden_dim], name="input2", is_input=True
    )
    down_block_res_samples_00 = Tensor(
        [batch_size, height, width, 320], name="input3", is_input=True
    )
    down_block_res_samples_01 = Tensor(
        [batch_size, height, width, 320], name="input4", is_input=True
    )
    down_block_res_samples_02 = Tensor(
        [batch_size, height, width, 320], name="input5", is_input=True
    )
    down_block_res_samples_03 = Tensor(
        [batch_size, height//2, width//2, 320], name="input6", is_input=True
    )
    down_block_res_samples_04 = Tensor(
        [batch_size, height//2, width//2, 640], name="input7", is_input=True
    )
    down_block_res_samples_05 = Tensor(
        [batch_size, height//2, width//2, 640], name="input8", is_input=True
    )
    down_block_res_samples_06 = Tensor(
        [batch_size, height//4, width//4, 640], name="input9", is_input=True
    )
    down_block_res_samples_07 = Tensor(
        [batch_size, height//4, width//4, 1280], name="input10", is_input=True
    )
    down_block_res_samples_08 = Tensor(
        [batch_size, height//4, width//4, 1280], name="input11", is_input=True
    )
    down_block_res_samples_09 = Tensor(
        [batch_size, height//8, width//8, 1280], name="input12", is_input=True
    )
    down_block_res_samples_10 = Tensor(
        [batch_size, height//8, width//8, 1280], name="input13", is_input=True
    )
    down_block_res_samples_11 = Tensor(
        [batch_size, height//8, width//8, 1280], name="input14", is_input=True
    )

    mid_block_res_sample = Tensor(
        [batch_size, height//8, width//8, 1280], name="input15", is_input=True
    )

    Y = ait_mod(latent_model_input_ait, timesteps_ait, text_embeddings_pt_ait,
                [down_block_res_samples_00,
                down_block_res_samples_01,
                down_block_res_samples_02,
                down_block_res_samples_03,
                down_block_res_samples_04,
                down_block_res_samples_05,
                down_block_res_samples_06,
                down_block_res_samples_07,
                down_block_res_samples_08,
                down_block_res_samples_09,
                down_block_res_samples_10,
                down_block_res_samples_11],
                mid_block_res_sample
                )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, ait_so_path, "UNet2DConditionModelControlNet", constants=None)
