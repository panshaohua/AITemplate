Stable Diffusion Controlnet Example


### Build AIT modules for CLIP, UNet, VAE

Build the AIT modules by running `compile_ait_controlnet_run.py`.

```
python examples/08_stable_diffusion_controlnet/scripts/compile_ait_controlnet_run.py
```
It generates three folders: `./controlnet_compile_lib/CLIPTextModel`, `./controlnet_compile_lib/UNet2DConditionModelControlNet`, `./controlnet_compile_lib/AutoencoderKL`, `./controlnet_compile_lib/ControlNetModel`. In each folder, there is a `test.so` file which is the generated AIT module for the model.


### Run Models

Run Torch models with an example image:
```
python examples/08_stable_diffusion_controlnet/scripts/demo_pt_controlnet.py
```


Run AIT models with an example image:

```
python examples/08_stable_diffusion_controlnet/scripts/demo_ait_controlnet.py
```