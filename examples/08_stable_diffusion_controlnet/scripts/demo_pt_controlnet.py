'''
 FILENAME:      demo_pt_controlnet.py

 AUTHORS:       Pan Shaohua

 START DATE:    Tuesday March 14th 2023

 CONTACT:       shaohua.pan@quvideo.com
'''

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import os
import cv2
import argparse
from PIL import Image
from datetime import datetime

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
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='a photo of woman wearing a suit, detailed face, detaile eyes, best quality, realistic',
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
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs/txt2img-images',
    )
    parser.add_argument(
        "--xformers", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--controlnet_cond_img_path", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=875884196
    )
    args = parser.parse_args()
    return args


def main(args):

    save_path = os.path.join(args.output_path, str(datetime.now()).split(' ')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # download an image
    image = load_image(args.controlnet_cond_img_path)
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_canny_model_path, 
        revision="fp16",
        torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model_path, 
        revision="fp16",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    # # speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # # remove following line if xformers is not installed
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()

    # generate image
    generator = torch.Generator(device='cuda')
    seed = args.seed
    generator.manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, device='cuda').to(torch.float16)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(5):
        
        start_event.record()
        
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps, 
            latents=latents, 
            image=canny_image,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale
        ).images[0]

        end_event.record()
        torch.cuda.synchronize()
        print(start_event.elapsed_time(end_event))

    image.save("{}/example_pt_controlnet_seed_{}.png".format(save_path, seed))


if __name__ == "__main__":
    args = parse_args()
    main(args)