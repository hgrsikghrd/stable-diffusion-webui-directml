import os
import json
import torch
import shutil
from typing import Union, Optional, Tuple, List
from pathlib import Path
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
)
from optimum.onnxruntime import ORTStableDiffusionXLPipeline
from olive.model import ONNXModel
from olive.workflows import run as olive_run

from modules import shared
from modules.paths_internal import sd_configs_path, models_path
from modules.sd_models import unload_model_weights
from modules.sd_onnx_utils import load_pipeline

available_sampling_methods = [
    "pndm",
    "lms",
    "heun",
    "euler",
    "euler-ancestral",
    "dpm",
    "ddim",
]


def ready(unoptimized_dir: str, optimized_dir: str):
    unload_model_weights()

    unoptimized_dir = Path(models_path) / "OliveCache" / unoptimized_dir
    optimized_dir = Path(models_path) / "ONNX-Olive" / optimized_dir

    return unoptimized_dir, optimized_dir


def cleanup(optimized_dir: str):
    if not shared.opts.cache_optimized_model:
        shutil.rmtree("cache", ignore_errors=True)
    shutil.rmtree("footprints", ignore_errors=True)
    shutil.rmtree(optimized_dir, ignore_errors=True)


def optimize_sd_from_ckpt(
    checkpoint: str,
    vae_id: str,
    vae_subfolder: str,
    unoptimized_dir: str,
    optimized_dir: str,
    safety_checker: bool,
    text_encoder: bool,
    unet: bool,
    vae_decoder: bool,
    vae_encoder: bool,
    scheduler_type: str,
    use_fp16: bool,
    sample_size: Union[Tuple[int, int], int],
    olive_merge_lora: bool,
    *olive_merge_lora_inputs,
):
    unoptimized_dir, optimized_dir = ready(unoptimized_dir, optimized_dir)
    shutil.rmtree(unoptimized_dir, ignore_errors=True)

    pipeline = StableDiffusionPipeline.from_single_file(
        os.path.join(models_path, "Stable-diffusion", checkpoint),
        torch_dtype=torch.float32,
        requires_safety_checker=False,
        scheduler_type=scheduler_type,
    )
    pipeline.save_pretrained(unoptimized_dir)
    del pipeline

    hw_synced = isinstance(sample_size, int)

    submodels = list()

    if safety_checker:
        submodels.append("safety_checker")
    if text_encoder:
        submodels.append("text_encoder")
    if unet:
        submodels.append("unet")
    if vae_decoder:
        submodels.append("vae_decoder")
    if vae_encoder:
        submodels.append("vae_encoder")

    optimize(
        unoptimized_dir,
        optimized_dir,
        True,
        vae_id,
        vae_subfolder,
        use_fp16,
        sample_size if hw_synced else sample_size[0],
        sample_size if hw_synced else sample_size[1],
        submodels,
        olive_merge_lora,
        *olive_merge_lora_inputs,
    )

    return ["Optimization complete."]


def optimize_sdxl_from_ckpt(
    checkpoint: str,
    vae_id: str,
    vae_subfolder: str,
    unoptimized_dir: str,
    optimized_dir: str,
    text_encoder: bool,
    text_encoder_2: bool,
    unet: bool,
    vae_decoder: bool,
    vae_encoder: bool,
    scheduler_type: str,
    use_fp16: bool,
    sample_size: Union[Tuple[int, int], int],
    olive_merge_lora: bool,
    *olive_merge_lora_inputs,
):
    unoptimized_dir, optimized_dir = ready(unoptimized_dir, optimized_dir)
    shutil.rmtree(unoptimized_dir, ignore_errors=True)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        os.path.join(models_path, "Stable-diffusion", checkpoint),
        torch_dtype=torch.float32,
        requires_safety_checker=False,
        scheduler_type=scheduler_type,
    )
    pipeline.save_pretrained(unoptimized_dir)
    del pipeline

    hw_synced = isinstance(sample_size, int)

    submodels = list()

    if text_encoder:
        submodels.append("text_encoder")
    if text_encoder_2:
        submodels.append("text_encoder_2")
    if unet:
        submodels.append("unet")
    if vae_decoder:
        submodels.append("vae_decoder")
    if vae_encoder:
        submodels.append("vae_encoder")

    optimize(
        unoptimized_dir,
        optimized_dir,
        True,
        vae_id,
        vae_subfolder,
        use_fp16,
        sample_size if hw_synced else sample_size[0],
        sample_size if hw_synced else sample_size[1],
        submodels,
        olive_merge_lora,
        *olive_merge_lora_inputs,
    )

    return ["Optimization complete."]


def optimize_sd_from_onnx(
    model_id: str,
    vae_id: str,
    vae_subfolder: str,
    unoptimized_dir: str,
    optimized_dir: str,
    safety_checker: bool,
    text_encoder: bool,
    unet: bool,
    vae_decoder: bool,
    vae_encoder: bool,
    use_fp16: bool,
    sample_size: Union[Tuple[int, int], int],
    olive_merge_lora: bool,
    *olive_merge_lora_inputs,
):
    unoptimized_dir, optimized_dir = ready(unoptimized_dir, optimized_dir)

    if os.path.isdir(unoptimized_dir):
        pipeline = load_pipeline(
            unoptimized_dir,
            False,
            torch_dtype=torch.float32,
            requires_safety_checker=False,
            local_files_only=True,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, requires_safety_checker=False
        )
        pipeline.save_pretrained(unoptimized_dir)
    del pipeline

    hw_synced = isinstance(sample_size, int)

    submodels = list()

    if safety_checker:
        submodels.append("safety_checker")
    if text_encoder:
        submodels.append("text_encoder")
    if unet:
        submodels.append("unet")
    if vae_decoder:
        submodels.append("vae_decoder")
    if vae_encoder:
        submodels.append("vae_encoder")

    optimize(
        unoptimized_dir,
        optimized_dir,
        True,
        vae_id,
        vae_subfolder,
        use_fp16,
        sample_size if hw_synced else sample_size[0],
        sample_size if hw_synced else sample_size[1],
        submodels,
        olive_merge_lora,
        *olive_merge_lora_inputs,
    )

    return ["Optimization complete."]


def optimize_sdxl_from_onnx(
    model_id: str,
    vae_id: str,
    vae_subfolder: str,
    unoptimized_dir: str,
    optimized_dir: str,
    text_encoder: bool,
    text_encoder_2: bool,
    unet: bool,
    vae_decoder: bool,
    vae_encoder: bool,
    use_fp16: bool,
    sample_size: Union[Tuple[int, int], int],
    olive_merge_lora: bool,
    *olive_merge_lora_inputs,
):
    unoptimized_dir, optimized_dir = ready(unoptimized_dir, optimized_dir)

    if os.path.isdir(unoptimized_dir):
        pipeline = load_pipeline(
            unoptimized_dir,
            True,
            torch_dtype=torch.float32,
            requires_safety_checker=False,
            local_files_only=True,
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, requires_safety_checker=False
        )
        pipeline.save_pretrained(unoptimized_dir)
    del pipeline

    hw_synced = isinstance(sample_size, int)

    submodels = list()

    if text_encoder:
        submodels.append("text_encoder")
    if text_encoder_2:
        submodels.append("text_encoder_2")
    if unet:
        submodels.append("unet")
    if vae_decoder:
        submodels.append("vae_decoder")
    if vae_encoder:
        submodels.append("vae_encoder")

    optimize(
        unoptimized_dir,
        optimized_dir,
        True,
        vae_id,
        vae_subfolder,
        use_fp16,
        sample_size if hw_synced else sample_size[0],
        sample_size if hw_synced else sample_size[1],
        submodels,
        olive_merge_lora,
        *olive_merge_lora_inputs,
    )

    return ["Optimization complete."]


def optimize(
    unoptimized_dir: Path,
    optimized_dir: Optional[Path],
    save_optimized: bool,
    vae_id: Optional[str],
    vae_subfolder: str,
    use_fp16: bool,
    sample_height: int,
    sample_width: int,
    submodels: List[str],
    olive_merge_lora: bool,
    *olive_merge_lora_inputs,
):
    if optimized_dir is not None:
        cleanup(optimized_dir)

    is_sdxl = "text_encoder_2" in submodels
    model_info = {}

    sample_height_dim = sample_height // 8
    sample_width_dim = sample_width // 8
    os.environ["OLIVE_IS_SDXL"] = str(int(is_sdxl))
    os.environ["OLIVE_CKPT_PATH"] = str(unoptimized_dir)
    os.environ["OLIVE_VAE"] = vae_id or str(unoptimized_dir)
    os.environ["OLIVE_VAE_SUBFOLDER"] = vae_subfolder
    os.environ["OLIVE_SAMPLE_HEIGHT_DIM"] = str(sample_height_dim)
    os.environ["OLIVE_SAMPLE_WIDTH_DIM"] = str(sample_width_dim)
    os.environ["OLIVE_SAMPLE_HEIGHT"] = str(sample_height)
    os.environ["OLIVE_SAMPLE_WIDTH"] = str(sample_width)
    os.environ["OLIVE_LORA_BASE_PATH"] = str(Path(models_path) / "Lora")
    if olive_merge_lora:
        os.environ["OLIVE_LORAS"] = "$".join(olive_merge_lora_inputs)

    for submodel_name in submodels:
        print(f"\nOptimizing {submodel_name}")

        with open(
            Path(sd_configs_path)
            / ("olive_optimize_sdxl" if is_sdxl else "olive_optimize")
            / f"config_{submodel_name}.json",
            "r",
        ) as olive_config_raw:
            olive_config = json.load(olive_config_raw)
        olive_config["passes"]["optimize"]["config"]["float16"] = use_fp16

        olive_run(olive_config)

        footprints_file_path = (
            Path("footprints") / f"{submodel_name}_gpu-dml_footprints.json"
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint

            assert conversion_footprint and optimizer_footprint

            unoptimized_olive_model = ONNXModel(
                **conversion_footprint["model_config"]["config"]
            )
            optimized_olive_model = ONNXModel(
                **optimizer_footprint["model_config"]["config"]
            )

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Optimized {submodel_name}")

    if save_optimized:
        print("Copying optimized models...")
        shutil.copytree(
            unoptimized_dir, optimized_dir, ignore=shutil.ignore_patterns("weights.pb", "*.safetensors", "*.ckpt")
        )

        pipeline = DiffusionPipeline.from_pretrained(unoptimized_dir)

        target_models = {"safety_checker": None}
        target_models["text_encoder"] = pipeline.text_encoder
        if hasattr(pipeline, "text_encoder_2"):
            target_models["text_encoder_2"] = pipeline.text_encoder_2
        target_models["vae_decoder"] = pipeline.vae_decoder if hasattr(pipeline, "vae_decoder") else pipeline.vae
        target_models["vae_encoder"] = pipeline.vae_encoder if hasattr(pipeline, "vae_encoder") else pipeline.vae

        other_models = {}
        other_models["tokenizer"] = pipeline.tokenizer
        if hasattr(pipeline, "tokenizer_2"):
            other_models["tokenizer_2"] = pipeline.tokenizer_2
        other_models["scheduler"] = pipeline.scheduler
        if hasattr(pipeline, "feature_extractor"):
            other_models["feature_extractor"] = pipeline.feature_extractor
        del pipeline

        for submodel in submodels:
            target_models[submodel] = OnnxRuntimeModel.from_pretrained(
                model_info[submodel]["unoptimized"]["path"].parent
            )

        pipeline = (
            ORTStableDiffusionXLPipeline(
                text_encoder_session=target_models["text_encoder"],
                text_encoder_2_session=target_models["text_encoder_2"],
                unet_session=target_models["unet"],
                vae_decoder_session=target_models["vae_decoder"],
                vae_encoder_session=target_models["vae_encoder"],
                **other_models,
                config=dict(pipeline.config),
            )
            if is_sdxl
            else OnnxStableDiffusionPipeline(
                **target_models,
                **other_models,
                requires_safety_checker=False,
            )
        )
        pipeline.to_json_file(optimized_dir / "model_index.json")
        del pipeline, target_models, other_models

        for submodel_name in submodels:
            try:
                src_path: Path = model_info[submodel_name]["optimized"]["path"]
                dst_path: Path = optimized_dir / submodel_name / "model.onnx"
                if not os.path.isdir(dst_path.parent):
                    os.mkdir(dst_path.parent)
                shutil.copyfile(src_path, dst_path)

                weights_src_path = src_path.parent / (src_path.name + ".data")
                if weights_src_path.is_file():
                    weights_dst_path = dst_path.parent / (dst_path.name + ".data")
                    shutil.copyfile(weights_src_path, weights_dst_path)
            except Exception:
                print(f"Error: Something went wrong. Failed to copy the component '{submodel_name}' of the optimized model.")

        with open(optimized_dir / "opt_config.json", "w") as opt_config:
            json.dump(
                {
                    "sample_height_dim": sample_height_dim,
                    "sample_width_dim": sample_width_dim,
                    "sample_height": sample_height,
                    "sample_width": sample_width,
                },
                opt_config,
            )

        shared.refresh_checkpoints()
    print("Optimization complete.")

    return model_info
