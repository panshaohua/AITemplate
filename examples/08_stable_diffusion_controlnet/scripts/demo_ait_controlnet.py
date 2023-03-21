'''
 FILENAME:      sd_ait_api.py

 AUTHORS:       Pan Shaohua

 START DATE:    Thursday January 5th 2023

 CONTACT:       shaohua.pan@quvideo.com
'''

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from collections import OrderedDict

from aitemplate.testing import detect_target
from diffusers.utils import load_image

import sys
sys.path.append('examples/08_stable_diffusion_controlnet')
from src.pipeline_stable_diffusion_controlnet_ait import StableDiffusionControlNetAITPipeline
from diffusers import ControlNetModel

# from src.modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
# from src.modeling.unet_2d_condition_with_controlnet import UNet2DConditionModel as ait_UNet2DConditionModel
# from src.modeling.vae import AutoencoderKL as ait_AutoencoderKL
# from src.modeling.controlnet import ControlNetModel as ait_ControlNetModel

USE_CUDA = detect_target().name() == "cuda"

def mark_output(y):
    if type(y) is not tuple and type(y) is not list:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))

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

def map_vae_params(ait_module, pt_module, batch_size, seq_len):
    pt_params = dict(pt_module.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_module.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            if (
                "conv" in name
                and "norm" not in name
                and name.endswith(".weight")
                and len(pt_params[name].shape) == 4
            ):
                mapped_pt_params[ait_name] = torch.permute(
                    pt_params[name], [0, 2, 3, 1]
                ).contiguous()
            else:
                mapped_pt_params[ait_name] = pt_params[name]
        elif name.endswith("attention.qkv.weight"):
            prefix = name[: -len("attention.qkv.weight")]
            q_weight = pt_params[prefix + "query.weight"]
            k_weight = pt_params[prefix + "key.weight"]
            v_weight = pt_params[prefix + "value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_pt_params[ait_name] = qkv_weight
        elif name.endswith("attention.qkv.bias"):
            prefix = name[: -len("attention.qkv.bias")]
            q_bias = pt_params[prefix + "query.bias"]
            k_bias = pt_params[prefix + "key.bias"]
            v_bias = pt_params[prefix + "value.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            mapped_pt_params[ait_name] = qkv_bias
        elif name.endswith("attention.proj.weight"):
            prefix = name[: -len("attention.proj.weight")]
            pt_name = prefix + "proj_attn.weight"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj.bias"):
            prefix = name[: -len("attention.proj.bias")]
            pt_name = prefix + "proj_attn.bias"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.cu_length"):
            cu_len = np.cumsum([0] + [seq_len] * batch_size).astype("int32")
            mapped_pt_params[ait_name] = torch.from_numpy(cu_len).cuda()
        else:
            pt_param = pt_module.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param

    return mapped_pt_params

def map_clip_params(pt_mod, batch_size, seqlen, depth):

    params_pt = list(pt_mod.named_parameters())

    params_ait = {}
    pt_params = {}
    for key, arr in params_pt:
        pt_params[key.replace("text_model.", "")] = arr

    pt_params = dict(pt_mod.named_parameters())
    for key, arr in pt_params.items():
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("q_proj.weight"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.weight")]
            q = pt_params[prefix + "q_proj.weight"]
            k = pt_params[prefix + "k_proj.weight"]
            v = pt_params[prefix + "v_proj.weight"]
            qkv_weight = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_weight
            continue
        elif name.endswith("q_proj.bias"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.bias")]
            q = pt_params[prefix + "q_proj.bias"]
            k = pt_params[prefix + "k_proj.bias"]
            v = pt_params[prefix + "v_proj.bias"]
            qkv_bias = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_bias
            continue
        elif name.endswith("k_proj.weight"):
            continue
        elif name.endswith("k_proj.bias"):
            continue
        elif name.endswith("v_proj.weight"):
            continue
        elif name.endswith("v_proj.bias"):
            continue
        params_ait[ait_name] = arr

        if USE_CUDA:
            for i in range(depth):
                prefix = "encoder_layers_%d_self_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait

def map_controlnet_params(pt_mod, dim):
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


def run(pretrained_model_path, prompt, num_inference_steps, out_image_numbers, ait_compiled_lib_path, output):

    controlnet = ControlNetModel.from_pretrained(
        "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
        revision="fp16",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionControlNetAITPipeline.from_pretrained(
        pretrained_model_path,
        controlnet=controlnet,
        revision="fp16",
        workdir=ait_compiled_lib_path,
        torch_dtype=torch.float16
    ).to("cuda")

    # ###########################################################
    # # clip_ait
    # mask_seq = 0
    # causal = True
    # depth = 12
    # dim=768
    # num_heads=12
    # batch_size=1
    # seqlen=64

    # ait_mod = ait_CLIPTextTransformer(
    #     num_hidden_layers=depth,
    #     hidden_size=dim,
    #     num_attention_heads=num_heads,
    #     batch_size=batch_size,
    #     seq_len=seqlen,
    #     causal=causal,
    #     mask_seq=mask_seq,
    # )
    # ait_mod.name_parameter_tensor()

    # pt_mod = pipe.text_encoder
    # pt_mod = pt_mod.eval()
    # params_ait = map_clip_params(pt_mod, batch_size, seqlen, depth)
    # for k, v in params_ait.items():
    #     pipe.clip_ait_exe.set_constant_with_tensor(k, v)

    # ###########################################################
    # # unet_ait
    # dim = 320
    # ait_mod = ait_UNet2DConditionModel(
    #     sample_size=64,
    #     cross_attention_dim=768
    # )
    # ait_mod.name_parameter_tensor()

    # # set AIT parameters
    # pt_mod = pipe.unet
    # pt_mod = pt_mod.eval()
    # params_ait = map_unet_params(pt_mod, dim)
    # for k, v in params_ait.items():
    #     pipe.unet_ait_exe.set_constant_with_tensor(k, v)

    # ###########################################################
    # # vae_ait
    # in_channels = 3
    # out_channels = 3
    # down_block_types = [
    #     "DownEncoderBlock2D",
    #     "DownEncoderBlock2D",
    #     "DownEncoderBlock2D",
    #     "DownEncoderBlock2D",
    # ]
    # up_block_types = [
    #     "UpDecoderBlock2D",
    #     "UpDecoderBlock2D",
    #     "UpDecoderBlock2D",
    #     "UpDecoderBlock2D",
    # ]
    # block_out_channels = [128, 256, 512, 512]
    # layers_per_block = 2
    # act_fn = "silu"
    # latent_channels = 4
    # sample_size = 512

    # ait_vae = ait_AutoencoderKL(
    #     batch_size,
    #     height,
    #     width,
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     down_block_types=down_block_types,
    #     up_block_types=up_block_types,
    #     block_out_channels=block_out_channels,
    #     layers_per_block=layers_per_block,
    #     act_fn=act_fn,
    #     latent_channels=latent_channels,
    #     sample_size=sample_size,
    # )
    # ait_vae.name_parameter_tensor()
    # # set AIT_VAE parameters
    # pt_mod = pipe.vae
    # pt_mod = pt_mod.eval()
    # params_ait = map_vae_params(pt_mod, dim)
    # for k, v in params_ait.items():
    #     pipe.vae_ait_exe.set_constant_with_tensor(k, v)
        
    # ###########################################################
    # # controlnet ait
    # ait_mod = ait_ControlNetModel(
    #     cross_attention_dim=768,
    # )
    # ait_mod.name_parameter_tensor()

    # # set AIT parameters
    # pt_mod = controlnet
    # pt_mod = pt_mod.eval()
    # params_ait = map_controlnet_params(pt_mod, dim)
    # for k, v in params_ait.items():
    #     pipe.clip_ait_exe.set_constant_with_tensor(k, v)
    # ###########################################################

    save_path = os.path.join(output_path, str(datetime.now()).split(' ')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = load_image('data/000006.jpg')
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    generator = torch.Generator(device='cuda')
    seed = 875884196
    generator.manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, device='cuda')
    num_inference_steps=20

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    ############################################################################
    with torch.autocast("cuda"):
        for i in range(out_image_numbers):

            start_event.record()

            image = pipe(
                prompt, 
                canny_image,
                512, 512, 
                latents=latents,
                num_inference_steps=num_inference_steps,
                negative_prompt='nsfw',
                controlnet_conditioning_scale=0.7,
                ).images[0]

            end_event.record()
            torch.cuda.synchronize()
            print(start_event.elapsed_time(end_event))

            image.save("{}/example_ait_controlnet_{}_{}.png".format(save_path, seed, i))
        

if __name__ == '__main__':
    pretrained_model_path = 'runwayml/stable-diffusion-v1-5'
    prompt = 'a photo of woman, beauty'
    num_inference_steps = 20
    output_path = 'outputs/txt2img-images'
    out_image_numbers = 5
    
    ait_compiled_lib_path = 'controlnet_compile_lib'

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(1):
        # start_event.record()
        run(pretrained_model_path, prompt, num_inference_steps, out_image_numbers, ait_compiled_lib_path, output_path)
        # end_event.record()
        # torch.cuda.synchronize()
        # print(start_event.elapsed_time(end_event))