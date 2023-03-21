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
import os
import unittest

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

try:
    from libfb.py.asyncio.await_utils import await_sync
    from manifold.clients.python import ManifoldClient
except ImportError:
    ManifoldClient = None

import sys
sys.path.append('examples/08_stable_diffusion_controlnet')
from src.benchmark import benchmark_clip, benchmark_unet, benchmark_vae
from src.compile_lib.compile_clip import compile_clip
from src.compile_lib.compile_unet_with_controlnet import compile_unet
from src.compile_lib.compile_vae import compile_vae


class StableDiffusionVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(StableDiffusionVerification, self).__init__(*args, **kwargs)

        self.local_path = "runwayml/stable-diffusion-v1-5"
        self.workdir = 'controlnet_compile_lib'

        try:
            controlnet = ControlNetModel.from_pretrained(
                "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
                revision="fp16",
                torch_dtype=torch.float16).to("cuda")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.local_path, revision="fp16", 
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to("cuda")
        except OSError:
            if ManifoldClient is not None:
                with ManifoldClient.get_client(bucket="glow_test_data") as client:
                    await_sync(
                        client.getRecursive(
                            manifold_path="tree/aitemplate/stable_diffusion/v2",
                            local_path=self.local_path,
                        )
                    )

                controlnet = ControlNetModel.from_pretrained(
                    "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
                    revision="fp16",
                    torch_dtype=torch.float16).to("cuda")
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    self.local_path, revision="fp16", 
                    controlnet=controlnet,
                    torch_dtype=torch.float16
                ).to("cuda")
            else:
                controlnet = ControlNetModel.from_pretrained(
                    "/mnt/users/shaohua/AIGC/A10/diffusers-pipeline/model_card/sd-controlnet-canny", 
                    revision="fp16",
                    torch_dtype=torch.float16).to("cuda")

                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=os.environ.get("HUGGINGFACE_AUTH_TOKEN", True),
                ).to("cuda")
                pipe.save_pretrained(self.local_path)

        self.pt_unet = pipe.unet
        self.pt_vae = pipe.vae
        self.pt_clip = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.pt_controlnet = pipe.controlnet

        self.vae_config = {
            "batch_size": 1,
            "width": 64,
            "height": 64,
        }

        self.unet_config = {
            "batch_size": 2,
            "dim": 320,
            "hidden_dim": pipe.unet.config.cross_attention_dim,
            "width": 64,
            "height": 64,
        }

        self.unet_compile_extra_config = {
            "attention_head_dim": pipe.unet.config.attention_head_dim,
        }

        self.clip_config = {
            "batch_size": 1,
            "seqlen": 64,
        }

        self.clip_compile_extra_config = {
            "depth": pipe.text_encoder.config.num_hidden_layers,
            "num_heads": pipe.text_encoder.config.num_attention_heads,
            "dim": pipe.text_encoder.config.hidden_size,
            "act_layer": pipe.text_encoder.config.hidden_act,
        }

    def test_vae(self):
        benchmark_vae(
            self.pt_vae,
            benchmark_pt=False,
            verify=True,
            **self.vae_config,
            ait_so_path=self.workdir
        )

    def test_unet(self):
        benchmark_unet(
            self.pt_unet,
            benchmark_pt=False,
            verify=True,
            **self.unet_config,
            ait_so_path=self.workdir
        )

    def test_clip(self):
        benchmark_clip(
            self.pt_clip,
            benchmark_pt=False,
            verify=True,
            tokenizer=self.tokenizer,
            **self.clip_config,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()