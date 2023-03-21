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
import inspect

import os
import warnings
from typing import List, Optional, Union

import numpy as np
import PIL.Image

import torch
from aitemplate.compiler import Model

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    ControlNetModel,
)

from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class StableDiffusionControlNetAITPipeline(StableDiffusionControlNetPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        workdir: str = 'tmp/'
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

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
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

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

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
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
            print(text_embeddings)
            input('text_embeddings')


        # 4. Prepare image
        cond_image = self.prepare_image(
            cond_image,
            width,
            height,
            batch_size,
            1,
            self.device,
            self.controlnet.dtype,
        )

        if do_classifier_free_guidance:
            cond_image = torch.cat([cond_image] * 2)

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
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

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
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
            # down_block_res_samples, mid_block_res_sample = self.controlnet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=text_embeddings,
            #     controlnet_cond=cond_image,
            #     return_dict=False,
            # )
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
            # noise_pred = self.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=text_embeddings,
            #     down_block_additional_residuals=down_block_res_samples,
            #     mid_block_additional_residual=mid_block_res_sample,
            # ).sample

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
                images=image, clip_input=safety_checker_input.pixel_values
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
