# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import os
import cv2
from PIL import Image
import click
from datetime import datetime

@click.command()
@click.option("--output", type=str, default='outputs/txt2img-images')
def run(output):

    save_path = os.path.join(output, str(datetime.now()).split(' ')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # download an image
    image = load_image('data/000006.jpg')
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
        revision="fp16",
        torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
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
    seed = 875884196
    generator.manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, device='cuda')
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.autocast("cuda"):
        for i in range(5):
            
            start_event.record()
            
            image = pipe(
                "a photo of woman, beauty", 
                negative_prompt='nsfw',
                num_inference_steps=20, 
                latents=latents, 
                image=canny_image,
                controlnet_conditioning_scale=0.7
            ).images[0]

            end_event.record()
            torch.cuda.synchronize()
            print(start_event.elapsed_time(end_event))

            image.save("{}/example_pt_controlnet_seed_{}_{}.png".format(save_path, seed, i))


if __name__ == "__main__":
    run()