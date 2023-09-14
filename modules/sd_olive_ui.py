import os
import json
import torch
import shutil
from typing import Union
from pathlib import Path
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from optimum.onnxruntime import ORTStableDiffusionXLPipeline
from olive.model import ONNXModel
from olive.workflows import run as olive_run

from modules import shared
from modules.paths_internal import sd_configs_path, models_path
from modules.sd_models import unload_model_weights
from modules.sd_onnx_utils import load_pipeline

available_sampling_methods = ["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm", "ddim"]

def optimize_from_ckpt(checkpoint: str, vae_id: str, vae_subfolder: str, unoptimized_dir: str, optimized_dir: str, safety_checker: bool, text_encoder: bool, text_encoder_2: bool, unet: bool, vae_decoder: bool, vae_encoder: bool, scheduler_type: str, use_fp16: bool, sample_size: Union[tuple[int, int], int], olive_merge_lora: bool, *olive_merge_lora_inputs):
    unload_model_weights()

    unoptimized_dir = Path(models_path) / "ONNX" / unoptimized_dir
    optimized_dir = Path(models_path) / "ONNX-Olive" / optimized_dir

    shutil.rmtree("footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_dir, ignore_errors=True)
    shutil.rmtree(optimized_dir, ignore_errors=True)

    pipeline = StableDiffusionPipeline.from_single_file(os.path.join(models_path, "Stable-diffusion", checkpoint), torch_dtype=torch.float32, requires_safety_checker=False, scheduler_type=scheduler_type)
    pipeline.save_pretrained(unoptimized_dir)

    hw_synced = isinstance(sample_size, int)

    optimize(unoptimized_dir, optimized_dir, pipeline, vae_id, vae_subfolder, safety_checker, text_encoder, text_encoder_2, unet, vae_decoder, vae_encoder, use_fp16, sample_size if hw_synced else sample_size[0], sample_size if hw_synced else sample_size[1], olive_merge_lora, *olive_merge_lora_inputs)

    return ["Optimization complete."]

def optimize_from_onnx(model_id: str, vae_id: str, vae_subfolder: str, unoptimized_dir: str, optimized_dir: str, safety_checker: bool, text_encoder: bool, text_encoder_2: bool, unet: bool, vae_decoder: bool, vae_encoder: bool, use_fp16: bool, sample_size: Union[tuple[int, int], int], olive_merge_lora: bool, *olive_merge_lora_inputs):
    unload_model_weights()

    unoptimized_dir = Path(models_path) / "ONNX" / unoptimized_dir
    optimized_dir = Path(models_path) / "ONNX-Olive" / optimized_dir

    shutil.rmtree("footprints", ignore_errors=True)
    shutil.rmtree(optimized_dir, ignore_errors=True)

    if os.path.isdir(unoptimized_dir):
        pipeline = load_pipeline(unoptimized_dir, text_encoder_2, torch_dtype=torch.float32, requires_safety_checker=False, local_files_only=True)
    else:
        pipeline = (StableDiffusionXLPipeline if text_encoder_2 else StableDiffusionPipeline).from_pretrained(model_id, torch_dtype=torch.float32, requires_safety_checker=False)
        pipeline.save_pretrained(unoptimized_dir)

    hw_synced = isinstance(sample_size, int)

    optimize(unoptimized_dir, optimized_dir, pipeline, vae_id, vae_subfolder, safety_checker, text_encoder, text_encoder_2, unet, vae_decoder, vae_encoder, use_fp16, sample_size if hw_synced else sample_size[0], sample_size if hw_synced else sample_size[1], olive_merge_lora, *olive_merge_lora_inputs)

    return ["Optimization complete."]

def optimize(unoptimized_dir: Path, optimized_dir: Path, pipeline, vae_id: str, vae_subfolder: str, safety_checker: bool, text_encoder: bool, text_encoder_2: bool, unet: bool, vae_decoder: bool, vae_encoder: bool, use_fp16: bool, sample_height: int, sample_width: int, olive_merge_lora: bool, *olive_merge_lora_inputs):
    model_info = {}
    submodels = []
    is_sdxl = text_encoder_2

    if safety_checker:
        submodels.append("safety_checker")
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
        os.environ["OLIVE_LORAS"] = '$'.join(olive_merge_lora_inputs)

    for submodel_name in submodels:
        print(f"\nOptimizing {submodel_name}")

        with open(Path(sd_configs_path) / ("olive_optimize_sdxl" if is_sdxl else "olive_optimize") / f"config_{submodel_name}.json", "r") as olive_config_raw:
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

            unoptimized_olive_model = ONNXModel(**conversion_footprint["model_config"]["config"])
            optimized_olive_model = ONNXModel(**optimizer_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Optimized {submodel_name}")

    print("\nCreating ONNX pipeline...")
    kwargs = dict()
    if is_sdxl:
        kwargs["text_encoder_2"] = None
    else:
        kwargs["safety_checker"] = None
    kwargs["text_encoder"] = None
    kwargs["unet"] = None
    kwargs["vae_decoder"] = None
    kwargs["vae_encoder"] = None
    for submodel in submodels:
        kwargs[submodel] = OnnxRuntimeModel.from_pretrained(model_info[submodel]["unoptimized"]["path"].parent)

    onnx_pipeline = ORTStableDiffusionXLPipeline(
        text_encoder_session=kwargs["text_encoder"],
        text_encoder_2_session=kwargs["text_encoder_2"],
        unet_session=kwargs["unet"],
        vae_decoder_session=kwargs["vae_decoder"],
        vae_encoder_session=kwargs["vae_encoder"],
        tokenizer=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        scheduler=pipeline.scheduler,
        feature_extractor=pipeline.feature_extractor,
        config=dict(pipeline.config),
    ) if is_sdxl else OnnxStableDiffusionPipeline(**kwargs,
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=False,
    )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_dir)

    print("Copying optimized models...")
    shutil.copytree(unoptimized_dir, optimized_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in submodels:
        try:
            src_path: Path = model_info[submodel_name]["optimized"]["path"]
            dst_path: Path = optimized_dir / submodel_name / "model.onnx"
            shutil.copyfile(src_path, dst_path)

            weights_src_path = src_path.parent / (src_path.name + ".data")
            if weights_src_path.is_file():
                weights_dst_path = dst_path.parent / (dst_path.name + ".data")
                shutil.copyfile(weights_src_path, weights_dst_path)
        except Exception:
            pass

    with open(optimized_dir / "opt_config.json", "w") as opt_config:
        json.dump({
            "sample_height_dim": sample_height_dim,
            "sample_width_dim": sample_width_dim,
            "sample_height": sample_height,
            "sample_width": sample_width,
        }, opt_config)

    shared.refresh_checkpoints()
    print("Optimization complete.")
