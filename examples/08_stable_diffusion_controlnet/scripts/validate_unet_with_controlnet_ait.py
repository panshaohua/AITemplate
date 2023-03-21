'''
 FILENAME:      demo_diffusers.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday December 16th 2022

 CONTACT:       shaohua.pan@quvideo.com
'''

import os
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from aitemplate.compiler import Model
import sys
sys.path.append('examples/05_stable_diffusion_controlnet')
from src.modeling.unet_2d_condition_with_controlnet import (
    UNet2DConditionModel as ait_UNet2DConditionModelWithControlNet,
)
from src.compile_lib.compile_controlnet import map_controlnet_params

torch.manual_seed(1234)

controlnet = ControlNetModel.from_pretrained(
    "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
    revision="fp16",
    torch_dtype=torch.float16).to("cuda:0")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

latent_model_input = torch.randn((2, 4, 64, 64), dtype=torch.float16).to('cuda:0')
# print(latent_model_input)
# input('latent_model_input')
t = torch.Tensor([999]).to('cuda:0')
prompt_embeds = torch.randn((2, 64, 768), dtype=torch.float16).to('cuda:0')

down_block_res_samples_00 = torch.randn((2, 320, 64, 64), dtype=torch.float16).to('cuda:0')
down_block_res_samples_01 = torch.randn((2, 320, 64, 64), dtype=torch.float16).to('cuda:0')
down_block_res_samples_02 = torch.randn((2, 320, 64, 64), dtype=torch.float16).to('cuda:0')
down_block_res_samples_03 = torch.randn((2, 320, 32, 32), dtype=torch.float16).to('cuda:0')
down_block_res_samples_04 = torch.randn((2, 640, 32, 32), dtype=torch.float16).to('cuda:0')
down_block_res_samples_05 = torch.randn((2, 640, 32, 32), dtype=torch.float16).to('cuda:0')
down_block_res_samples_06 = torch.randn((2, 640, 16, 16), dtype=torch.float16).to('cuda:0')
down_block_res_samples_07 = torch.randn((2, 1280, 16, 16), dtype=torch.float16).to('cuda:0')
down_block_res_samples_08 = torch.randn((2, 1280, 16, 16), dtype=torch.float16).to('cuda:0')
down_block_res_samples_09 = torch.randn((2, 1280, 8, 8), dtype=torch.float16).to('cuda:0')
down_block_res_samples_10 = torch.randn((2, 1280, 8, 8), dtype=torch.float16).to('cuda:0')
down_block_res_samples_11 = torch.randn((2, 1280, 8, 8), dtype=torch.float16).to('cuda:0')

mid_block_res_sample = torch.randn((2, 1280, 8, 8), dtype=torch.float16).to('cuda:0')

out = pipe.unet(
    latent_model_input,
    t,
    encoder_hidden_states=prompt_embeds,
    down_block_additional_residuals=[
        down_block_res_samples_00,
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
        down_block_res_samples_11
    ],
    mid_block_additional_residual=mid_block_res_sample,
    return_dict=False,
)


# print(latent_model_input)
print(out)
# print(prompt_embeds)
# print(image)

# torch.Size([2, 4, 64, 64]) 
# torch.Size([])
# torch.Size([2, 77, 768])
# torch.Size([2, 3, 512, 512])

def init_ait_module(
    model_name,
    workdir,
    ):
    mod = Model(os.path.join(workdir, model_name, "test.so"))
    return mod


def unet_with_controlnet_inference(latent_model_input, timesteps, encoder_hidden_states, 
                                   down_block_additional_residuals,
                                   mid_block_additional_residual):
    exe_module = unet_with_controlnet_ait_exe
    timesteps_pt = timesteps.expand(latent_model_input.shape[0])
    inputs = {
        "input0": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input1": timesteps_pt.cuda().half(),
        "input2": encoder_hidden_states.cuda().half(),
        "input3": down_block_additional_residuals[0].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input4": down_block_additional_residuals[1].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input5": down_block_additional_residuals[2].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input6": down_block_additional_residuals[3].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input7": down_block_additional_residuals[4].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input8": down_block_additional_residuals[5].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input9": down_block_additional_residuals[6].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input10": down_block_additional_residuals[7].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input11": down_block_additional_residuals[8].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input12": down_block_additional_residuals[9].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input13": down_block_additional_residuals[10].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input14": down_block_additional_residuals[11].permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input15": mid_block_additional_residual.permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half()
    }
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    pred = ys[0].permute((0, 3, 1, 2)).float()
    return pred



unet_with_controlnet_ait_exe = init_ait_module(
    model_name="UNet2DConditionModelControlNet", workdir='tmp/'
)
ait_mod = ait_UNet2DConditionModelWithControlNet(
    cross_attention_dim=768,
    attention_head_dim=8,
)
ait_mod.name_parameter_tensor()
pt_mod = pipe.unet
pt_mod = pt_mod.eval()
dim=320
params_ait = map_controlnet_params(pt_mod, dim)
for k, v in params_ait.items():
    unet_with_controlnet_ait_exe.set_constant_with_tensor(k, v)

pred = unet_with_controlnet_inference(latent_model_input, t, prompt_embeds,
                                      [
                                        down_block_res_samples_00,
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
                                        down_block_res_samples_11
                                    ],
                                      mid_block_res_sample)
print('......')