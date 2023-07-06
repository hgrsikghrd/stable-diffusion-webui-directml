# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import torch
import safetensors.torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import model_info
from transformers.models.clip.modeling_clip import CLIPTextModel


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label


def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model", model_name)


def is_lora_model(model_name):
    # TODO: might be a better way to detect (e.g. presence of LORA weights file)
    return model_name != get_base_model_name(model_name)


# Merges LoRA weights into the layers of a base model
def merge_lora_weights(text_encoder: CLIPTextModel, unet: UNet2DConditionModel, lora_model_path: str, ratio=1.0):
    # Load LoRA weights
    if lora_model_path.split('.')[-1].lower() == "safetensors":
        lora_state_dict = safetensors.torch.load_file(lora_model_path, device="cpu")
    else:
        lora_state_dict = torch.load(lora_model_path, map_location="cpu")

    """
    Merging LoRA
    kohya-ss/sd-scripts
    networks/merge_lora.py 37-102
    https://github.com/kohya-ss/sd-scripts/blob/main/networks/merge_lora.py
    http://www.apache.org/licenses/LICENSE-2.0
    Copyright [2022] [kohya-ss]
    """
    for key in list(lora_state_dict.keys()):
        if type(lora_state_dict[key]) == torch.Tensor:
            lora_state_dict[key] = lora_state_dict[key].to(unet.dtype)

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = "lora_te"
            target_replace_modules = ["CLIPAttention", "CLIPMLP"]
        else:
            prefix = "lora_unet"
            target_replace_modules = ["Transformer2DModel", "Attention", "ResnetBlock2D", "Downsample2D", "Upsample2D"]

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    for key in lora_state_dict.keys():
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            # find original module for this lora
            module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
            if module_name not in name_to_module:
                print(f"no module found for LoRA weight: {key}")
                continue
            module = name_to_module[module_name]
            # print(f"apply {key} to {module}")

            down_weight = lora_state_dict[key]
            up_weight = lora_state_dict[up_key]

            dim = down_weight.size()[0]
            alpha = lora_state_dict.get(alpha_key, dim)
            scale = alpha / dim

            # W <- W + U * D
            weight = module.weight
            # print(module_name, down_weight.size(), up_weight.size())
            if len(weight.size()) == 2:
                # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # print(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + ratio * conved * scale

            module.weight = torch.nn.Parameter(weight)


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    checkpoint_path = os.environ.get("OLIVE_CKPT_PATH")
    lora_str = os.environ.get("OLIVE_LORAS")
    model = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
    if lora_str is not None:
        loras: list[str] = lora_str.split('$')
        unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
        for lora in loras:
            if lora:
                filename = lora.split('\\')[-1]
                print(f"Merging LoRA {filename}...")
                merge_lora_weights(model, unet, os.path.join(os.environ.get("OLIVE_LORA_BASE_PATH"), lora))
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 4, int(os.environ.get("OLIVE_SAMPLE_HEIGHT_DIM", 64)), int(os.environ.get("OLIVE_SAMPLE_WIDTH_DIM", 64))), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 768), dtype=torch_dtype),
        "return_dict": False,
    }


def unet_load(model_name):
    checkpoint_path = os.environ.get("OLIVE_CKPT_PATH")
    lora_str = os.environ.get("OLIVE_LORAS")
    model = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
    if lora_str is not None:
        loras: list[str] = lora_str.split('$')
        text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
        for lora in loras:
            if lora:
                filename = lora.split('\\')[-1]
                print(f"Merging LoRA {filename}...")
                merge_lora_weights(text_encoder, model, os.path.join(os.environ.get("OLIVE_LORA_BASE_PATH"), lora))
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32).values())


def unet_data_loader(data_dir, batchsize):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, int(os.environ.get("OLIVE_SAMPLE_HEIGHT", 512)), int(os.environ.get("OLIVE_SAMPLE_WIDTH", 512))), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    subfolder = os.environ.get("OLIVE_VAE_SUBFOLDER") or None
    model = AutoencoderKL.from_pretrained(os.environ.get("OLIVE_VAE"), subfolder=subfolder)
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, int(os.environ.get("OLIVE_SAMPLE_HEIGHT_DIM", 64)), int(os.environ.get("OLIVE_SAMPLE_WIDTH_DIM", 64))), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    subfolder = os.environ.get("OLIVE_VAE_SUBFOLDER") or None
    model = AutoencoderKL.from_pretrained(os.environ.get("OLIVE_VAE"), subfolder=subfolder)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batchsize, torch_dtype):
    return {
        "clip_input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batchsize, int(os.environ.get("OLIVE_SAMPLE_HEIGHT", 512)), int(os.environ.get("OLIVE_SAMPLE_WIDTH", 512)), 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    model = StableDiffusionSafetyChecker.from_pretrained(os.environ.get("OLIVE_CKPT_PATH"), subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model


def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())


def safety_checker_data_loader(data_dir, batchsize):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)
