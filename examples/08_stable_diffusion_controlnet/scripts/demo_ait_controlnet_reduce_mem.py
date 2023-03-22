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
import argparse
from datetime import datetime
from PIL import Image
from collections import OrderedDict

from aitemplate.testing import detect_target

import inspect
from typing import List, Optional, Union
import PIL.Image
from aitemplate.compiler import Model
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from diffusers import LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from tqdm.auto import tqdm


USE_CUDA = detect_target().name() == "cuda"

class AITControlNetInfer(object):
    def __init__(self, args):
        workdir = args.compiled_ait_lib_path

        self.clip_ait_exe = self.init_ait_module(
            model_name="CLIPTextModel", workdir=workdir
        )
        self.unet_ait_exe = self.init_ait_module(
            model_name="UNet2DConditionModelControlNet", workdir=workdir
        )
        self.vae_ait_exe = self.init_ait_module(
            model_name="AutoencoderKL", workdir=workdir
        )
        self.controlnet_ait_exe = self.init_ait_module(
            model_name="ControlNetModel", workdir=workdir
        )

        self.device = args.device

        self.tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(args.base_model_path, "feature_extractor", "preprocessor_config.json"))
        self.scheduler = PNDMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler", torch_dtype=torch.float16)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(args.base_model_path, subfolder="safety_checker", torch_dtype=torch.float16).to(self.device)

    def init_ait_module(
        self,
        model_name,
        workdir,
    ):
        mod = Model(os.path.join(workdir, model_name, "test.so"))
        return mod

    def unet_inference(self, latent_model_input, timesteps, encoder_hidden_states,
                       down_block_additional_residuals,
                       mid_block_additional_residual):
        exe_module = self.unet_ait_exe
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
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=64):
        exe_module = self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out

    def controlnet_inference(self, latent_model_input, timesteps, encoder_hidden_states, controlnet_cond):
        exe_module = self.controlnet_ait_exe
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
        
        pred = []
        for i in range(num_outputs):
            pred.append(ys[i].permute((0, 3, 1, 2)).float())
        return pred

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL.Image.LANCZOS))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        cond_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        controlnet_conditioning_scale: float = 1.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=64,  # self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))

        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 4. Prepare image
        cond_image = self.prepare_image(
            cond_image,
            width,
            height,
            batch_size,
            1,
            self.device,
            torch.float16,
        )

        if do_classifier_free_guidance:
            cond_image = torch.cat([cond_image] * 2)

        # get the initial random noise unless the user supplied it
        latents_device = "cpu" if self.device.type == "mps" else self.device
        # self.unet.in_channels: 4 TODO
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[i]
                    # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # controlnet 
                controlnet_output = self.controlnet_inference(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=cond_image
                )
                down_block_res_samples, mid_block_res_sample = controlnet_output[0:12], controlnet_output[-1]

                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale

                # predict the noise residual
                noise_pred = self.unet_inference(
                    latent_model_input, t, encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(
                        noise_pred, i, latents, **extra_step_kwargs
                    ).prev_sample
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample

                if i == len(self.scheduler.timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_inference(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(torch.float16)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


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


def main(args):


    InferEngine = AITControlNetInfer(args)

    # ###########################################################
    # setup parameter

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

    save_path = os.path.join(args.output_path, str(datetime.now()).split(' ')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = PIL.Image.open(args.controlnet_cond_img_path)
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    generator = torch.Generator(device='cuda')
    seed = args.seed
    generator.manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, device='cuda:0').to(torch.float16)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    ############################################################################
    for i in range(3):

        start_event.record()

        image = InferEngine(
            args.prompt, 
            canny_image,
            512, 512, 
            latents=latents,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            ).images[0]

        end_event.record()
        torch.cuda.synchronize()
        print(start_event.elapsed_time(end_event))

        image.save("{}/example_ait_controlnet_{}_{}.png".format(save_path, seed, i))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_canny_model_path",
        type=str,
        default='lllyasviel/sd-controlnet-canny',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='a photo of woman wearing a suit, detailed face, detaile eyes, best quality, realistic',
        # required=True,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default='NSFW',
        # required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs/txt2img-images',
        # required=True,
    )
    parser.add_argument(
        "--xformers", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--compiled_ait_lib_path", 
        type=str,
        default="controlnet_compile_lib"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--controlnet_cond_img_path", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=875884196
    )
    parser.add_argument(
        "--device",
        default=torch.device('cuda', 0)
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)