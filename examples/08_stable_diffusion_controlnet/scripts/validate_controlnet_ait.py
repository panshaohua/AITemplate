'''
 FILENAME:      demo_diffusers.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday December 16th 2022

 CONTACT:       shaohua.pan@quvideo.com
'''

import os
import torch
from diffusers import ControlNetModel

from aitemplate.compiler import Model
import sys
sys.path.append('examples/05_stable_diffusion_controlnet')
from src.modeling.controlnet import (
    ControlNetModel as ait_ControlNetModel,
)
from src.compile_lib.compile_controlnet import map_controlnet_params

torch.manual_seed(1234)

controlnet = ControlNetModel.from_pretrained(
    "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
    revision="fp16",
    torch_dtype=torch.float16).to("cuda:0")

latent_model_input = torch.randn((2, 4, 64, 64), dtype=torch.float16).to('cuda:0')
# print(latent_model_input)
# input('latent_model_input')
t = torch.Tensor([999]).to('cuda:0')
prompt_embeds = torch.randn((2, 64, 768), dtype=torch.float16).to('cuda:0')
image = torch.randn((2, 3, 512, 512), dtype=torch.float16).to('cuda:0')

out = controlnet(
    latent_model_input,
    t,
    encoder_hidden_states=prompt_embeds,
    controlnet_cond=image,
    return_dict=False,
)


# print(latent_model_input)
print(t)
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


def controlnet_inference(latent_model_input, timesteps, encoder_hidden_states, controlnet_cond):
    exe_module = controlnet_ait_exe
    timesteps_pt = timesteps.expand(latent_model_input.shape[0])
    inputs = {
        "input0": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input1": timesteps_pt.cuda().half(),
        "input2": encoder_hidden_states.cuda().half(),
        "input3": controlnet_cond.permute((0, 2, 3, 1))
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



controlnet_ait_exe = init_ait_module(
    model_name="ControlNetModel", workdir='tmp/'
)
ait_mod = ait_ControlNetModel(
    cross_attention_dim=768,
    attention_head_dim=8,
)
ait_mod.name_parameter_tensor()
pt_mod = controlnet
pt_mod = pt_mod.eval()
dim=320
params_ait = map_controlnet_params(pt_mod, dim)
for k, v in params_ait.items():
    controlnet_ait_exe.set_constant_with_tensor(k, v)

pred = controlnet_inference(latent_model_input, t, prompt_embeds, image)
print('......')