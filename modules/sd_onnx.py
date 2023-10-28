import gc
import sys
import cv2
import torch
import numpy as np
import hashlib
import os.path
import diffusers
from transformers import CLIPTokenizer, CLIPImageProcessor
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TypeVar, Generic, Union, Any
from pathlib import Path
from PIL import Image, ImageOps
import onnxruntime as ort

from modules import shared, images, devices, masking, processing
from modules.sd_models_types import WebuiSdModel
from modules.sd_models import CheckpointInfo
from modules.sd_onnx_hijack import do_hijack
from modules.sd_samplers import find_sampler_config
from modules.sd_samplers_common import SamplerData

do_hijack()

device_map = defaultdict(lambda: ("DmlExecutionProvider", {
    "device_id": int(shared.cmd_opts.device_id or 0)
}))

T2I = TypeVar("T2I")
I2I = TypeVar("I2I")
INP = TypeVar("INP")


class BaseONNXModel(Generic[T2I, I2I, INP], WebuiSdModel, metaclass=ABCMeta):
    _sess_options: ort.SessionOptions

    filename: str
    is_sd1 = False
    is_sd2 = False
    is_sdxl: bool = False
    is_onnx: bool = True
    is_optimized: bool = False
    path: Path
    sd_model_hash: None = None
    sd_checkpoint_info: CheckpointInfo
    cond_stage_model: torch.nn.Module = torch.nn.Module()
    cond_stage_key: str = ""
    pipeline: Union[diffusers.DiffusionPipeline, None] = None
    vae: None = None

    dtype: torch.dtype = devices.dtype
    device: torch.device = devices.device

    def __init__(self, ckpt_info: CheckpointInfo):
        self.sd_checkpoint_info = ckpt_info
        self.filename = ckpt_info.filename
        self.path = Path(ckpt_info.filename)
        self.is_optimized = ckpt_info.is_optimized
        self._sess_options = ort.SessionOptions()

    def to(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.dtype = arg
            if isinstance(arg, torch.device):
                self.device = arg

        for key in kwargs:
            if key == "dtype":
                self.dtype = kwargs["dtype"]
            if key == "device":
                self.device = kwargs["device"]

        return self

    def add_free_dimension_override_by_name(self, *args, **kwargs):
        return self._sess_options.add_free_dimension_override_by_name(*args, **kwargs)

    def set_mem_pattern(self, enabled: bool):
        self._sess_options.enable_mem_pattern = enabled

    def set_mem_reuse(self, enabled: bool):
        self._sess_options.enable_mem_reuse = enabled

    def get_pipeline_config(self) -> dict:
        return {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "offload_state_dict": shared.opts.offload_state_dict,
        }

    def load_orm(self, submodel: str) -> Union[diffusers.OnnxRuntimeModel, None]:
        path = self.path / submodel
        if not self.sd_checkpoint_info.is_optimized and self.sd_checkpoint_info.optimized_model_info is not None:
            path = self.sd_checkpoint_info.optimized_model_info[submodel]["optimized"]["path"].parent
        return diffusers.OnnxRuntimeModel.from_pretrained(
            path, provider=device_map[submodel]
        )

    def load_inference_session(self, submodel: str) -> Union[ort.InferenceSession, None]:
        path = self.path / submodel / "model.onnx"
        if not self.sd_checkpoint_info.is_optimized and self.sd_checkpoint_info.optimized_model_info is not None:
            path = self.sd_checkpoint_info.optimized_model_info[submodel]["optimized"]["path"]
        return diffusers.OnnxRuntimeModel.load_model(
            path, provider=device_map[submodel]
        )

    def load_tokenizer(self, name: str) -> Union[CLIPTokenizer, None]:
        return (
            CLIPTokenizer.from_pretrained(self.path / name)
            if os.path.isdir(self.path / name)
            else None
        )

    def load_image_processor(self, name: str) -> Union[CLIPImageProcessor, None]:
        return (
            CLIPImageProcessor.from_pretrained(self.path / name)
            if os.path.isdir(self.path / name)
            else None
        )

    def create_pipeline(self, processing, sampler: SamplerData) -> Union[T2I, I2I]:
        if isinstance(processing, ONNXStableDiffusionProcessingTxt2Img):
            return self.create_txt2img_pipeline(sampler)
        elif isinstance(processing, ONNXStableDiffusionProcessingImg2Img):
            if processing.image_mask is None:
                return self.create_img2img_pipeline(sampler)
            else:
                return self.create_inpaint_pipeline(sampler)

    @abstractmethod
    def create_txt2img_pipeline(self, sampler: SamplerData) -> T2I:
        pass

    @abstractmethod
    def create_img2img_pipeline(self, sampler: SamplerData) -> I2I:
        pass

    @abstractmethod
    def create_inpaint_pipeline(self, sampler: SamplerData) -> INP:
        pass


@dataclass(repr=False)
class ONNXStableDiffusionProcessing(metaclass=ABCMeta):
    sd_model: BaseONNXModel = None
    outpath_samples: str = None
    outpath_grids: str = None
    prompt: str = ""
    prompt_for_display: str = None
    negative_prompt: str = ""
    styles: list[str] = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    restore_faces: bool = None
    tiling: bool = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: dict[str, Any] = None
    overlay_images: list = None
    eta: float = None
    do_not_reload_embeddings: bool = False
    denoising_strength: float = 0
    ddim_discretize: str = None
    s_min_uncond: float = None
    s_churn: float = None
    s_tmax: float = None
    s_tmin: float = None
    s_noise: float = None
    override_settings: dict[str, Any] = None
    override_settings_restore_afterwards: bool = True
    sampler_index: int = None
    refiner_checkpoint: str = None
    refiner_switch_at: float = None
    token_merging_ratio = 0
    token_merging_ratio_hr = 0
    disable_extra_networks: bool = False

    scripts_value: processing.scripts.ScriptRunner = field(default=None, init=False)
    script_args_value: list = field(default=None, init=False)
    scripts_setup_complete: bool = field(default=False, init=False)

    cached_uc = [None, None]
    cached_c = [None, None]

    comments: dict = None
    sampler: SamplerData | None = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: tuple | None = field(default=None, init=False)

    is_hr_pass: bool = field(default=False, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

    rng: processing.rng.ImageRNG | None = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
    color_corrections: list = field(default=None, init=False)

    all_prompts: list = field(default=None, init=False)
    all_negative_prompts: list = field(default=None, init=False)
    all_seeds: list = field(default=None, init=False)
    all_subseeds: list = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: str = field(default=None, init=False)
    main_negative_prompt: str = field(default=None, init=False)

    prompts: list = field(default=None, init=False)
    negative_prompts: list = field(default=None, init=False)
    seeds: list = field(default=None, init=False)
    subseeds: list = field(default=None, init=False)
    extra_network_data: dict = field(default=None, init=False)

    user: str = field(default=None, init=False)

    sd_model_name: str = field(default=None, init=False)
    sd_model_hash: str = field(default=None, init=False)
    sd_vae_name: str = field(default=None, init=False)
    sd_vae_hash: str = field(default=None, init=False)

    is_api: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.sampler_index is not None:
            print(
                "sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name",
                file=sys.stderr,
            )

        self.comments = {}

        if self.styles is None:
            self.styles = []

        self.sampler_noise_scheduler_override = None
        self.s_min_uncond = (
            self.s_min_uncond
            if self.s_min_uncond is not None
            else shared.opts.s_min_uncond
        )
        self.s_churn = self.s_churn if self.s_churn is not None else shared.opts.s_churn
        self.s_tmin = self.s_tmin if self.s_tmin is not None else shared.opts.s_tmin
        self.s_tmax = (
            self.s_tmax if self.s_tmax is not None else shared.opts.s_tmax
        ) or float("inf")
        self.s_noise = self.s_noise if self.s_noise is not None else shared.opts.s_noise

        self.extra_generation_params = self.extra_generation_params or {}
        self.override_settings = self.override_settings or {}
        self.script_args = self.script_args or {}

        self.refiner_checkpoint_info = None

        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.cached_uc = processing.StableDiffusionProcessing.cached_uc
        self.cached_c = processing.StableDiffusionProcessing.cached_c

        self.sampler = find_sampler_config(self.sampler_name)
        if self.sampler is None:
            raise Exception("Unknown sampler.")

        if not self.sd_model.is_optimized and self.sd_model.sd_checkpoint_info.optimized_model_info is None:
            from modules.sd_olive_ui import optimize
            self.sd_model.sd_checkpoint_info.optimized_model_info = optimize(
                unoptimized_dir=self.sd_model.path,
                optimized_dir=None,
                save_optimized=False,
                vae_id=None,
                vae_subfolder="vae",
                use_fp16=True,
                sample_height=self.height,
                sample_width=self.width,
                submodels=[
                    "text_encoder",
                    "unet",
                    "vae_decoder",
                    "vae_encoder",
                ] + (
                    ["text_encoder_2"] if self.sd_model.is_sdxl else []
                ),
                olive_merge_lora=False,
            )

        self.sd_model.set_mem_pattern(shared.opts.enable_mem_pattern)
        self.sd_model.set_mem_reuse(shared.opts.enable_mem_reuse)
        self.sd_model.add_free_dimension_override_by_name(
            "unet_sample_batch", self.batch_size * 2
        )
        self.sd_model.add_free_dimension_override_by_name(
            "unet_hidden_batch", self.batch_size * 2
        )
        if self.sd_model.is_sdxl:
            self.sd_model.add_free_dimension_override_by_name(
                "unet_text_embeds_batch", self.batch_size * 2
            )
            self.sd_model.add_free_dimension_override_by_name(
                "unet_text_embeds_size", self.height + 256
            )
            self.sd_model.add_free_dimension_override_by_name(
                "unet_time_ids_batch", self.batch_size * 2
            )

    @property
    def scripts(self):
        return self.scripts_value

    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value

        if (
            self.scripts_value
            and self.script_args_value
            and not self.scripts_setup_complete
        ):
            self.setup_scripts()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value

        if (
            self.scripts_value
            and self.script_args_value
            and not self.scripts_setup_complete
        ):
            self.setup_scripts()

    def setup_scripts(self):
        self.scripts_setup_complete = True

        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    def setup_prompts(self):
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            self.all_prompts = self.batch_size * [self.prompt]

        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(
                f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})"
            )

        self.all_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, self.styles)
            for x in self.all_prompts
        ]
        self.all_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles)
            for x in self.all_negative_prompts
        ]

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def _create_pipeline(self):
        return self.sd_model.create_pipeline(self, self.sampler)

    @property
    def pipeline(self) -> diffusers.DiffusionPipeline:
        if shared.opts.collect_garbage_for_each_generation:
            self.sd_model.pipeline = None
        return self.sd_model.pipeline or self._create_pipeline()

    @abstractmethod
    def forward(self) -> processing.Processed:
        pass

    def __call__(self) -> processing.Processed:
        return self.forward()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        if not shared.opts.persistent_cond_cache:
            processing.StableDiffusionProcessing.cached_c = [None, None]
            processing.StableDiffusionProcessing.cached_uc = [None, None]


@dataclass(repr=False)
class ONNXStableDiffusionProcessingTxt2Img(ONNXStableDiffusionProcessing):
    enable_hr: bool = False
    denoising_strength: float = 0.75
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: float = 2.0
    hr_upscaler: str = None
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: str = None
    hr_sampler_name: str = None
    hr_prompt: str = ""
    hr_negative_prompt: str = ""

    cached_hr_uc = [None, None]
    cached_hr_c = [None, None]

    hr_checkpoint_info: dict = field(default=None, init=False)
    hr_upscale_to_x: int = field(default=0, init=False)
    hr_upscale_to_y: int = field(default=0, init=False)
    truncate_x: int = field(default=0, init=False)
    truncate_y: int = field(default=0, init=False)
    applied_old_hires_behavior_to: tuple = field(default=None, init=False)
    latent_scale_mode: dict = field(default=None, init=False)
    hr_c: tuple | None = field(default=None, init=False)
    hr_uc: tuple | None = field(default=None, init=False)
    all_hr_prompts: list = field(default=None, init=False)
    all_hr_negative_prompts: list = field(default=None, init=False)
    hr_prompts: list = field(default=None, init=False)
    hr_negative_prompts: list = field(default=None, init=False)
    hr_extra_network_data: list = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.firstphase_width != 0 or self.firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = self.firstphase_width
            self.height = self.firstphase_height

        self.cached_hr_uc = processing.StableDiffusionProcessingTxt2Img.cached_hr_uc
        self.cached_hr_c = processing.StableDiffusionProcessingTxt2Img.cached_hr_c

    def setup_prompts(self):
        super().setup_prompts()

        if not self.enable_hr:
            return

        if self.hr_prompt == "":
            self.hr_prompt = self.prompt

        if self.hr_negative_prompt == "":
            self.hr_negative_prompt = self.negative_prompt

        if isinstance(self.hr_prompt, list):
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * [self.hr_prompt]

        if isinstance(self.hr_negative_prompt, list):
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * [self.hr_negative_prompt]

        self.all_hr_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, self.styles)
            for x in self.all_hr_prompts
        ]
        self.all_hr_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles)
            for x in self.all_hr_negative_prompts
        ]

    def forward(self) -> processing.Processed:
        if isinstance(self.prompt, list):
            assert len(self.prompt) > 0
        else:
            assert self.prompt is not None

        gc.collect()
        torch.cuda.empty_cache()

        seed = processing.get_fixed_seed(self.seed)
        subseed = processing.get_fixed_seed(self.subseed)

        if isinstance(seed, list):
            self.all_seeds = seed
        else:
            self.all_seeds = [
                int(seed) + (x if self.subseed_strength == 0 else 0)
                for x in range(len(self.all_prompts))
            ]

        if isinstance(subseed, list):
            self.all_subseeds = subseed
        else:
            self.all_subseeds = [int(subseed) + x for x in range(len(self.all_prompts))]

        if shared.state.job_count == -1:
            shared.state.job_count = self.n_iter * self.steps

        output_images = []

        for i in range(0, self.n_iter):
            result: diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput = self.pipeline(
                self,
                prompt=self.all_prompts,
                negative_prompt=self.all_negative_prompts,
                num_inference_steps=self.steps,
                num_images_per_prompt=1,
                height=self.height,
                width=self.width,
                eta=self.eta,
                seed=seed + i,
            )
            for image in result.images:
                images.save_image(image, self.outpath_samples, "")
            output_images += result.images

            result.images = None
            result = None
            gc.collect()
            torch.cuda.empty_cache()

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = (
            len(output_images) < 2 and shared.opts.grid_only_if_multiple
        )
        if (
            shared.opts.return_grid or shared.opts.grid_save
        ) and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, self.n_iter)

            if shared.opts.return_grid:
                output_images.insert(0, grid)
                index_of_first_image = 1

            if shared.opts.grid_save:
                images.save_image(
                    grid,
                    self.outpath_grids,
                    "grid",
                    self.all_seeds[0],
                    self.all_prompts[0],
                    shared.opts.grid_format,
                    short_filename=not shared.opts.grid_extended_filename,
                    grid=True,
                )

        gc.collect()
        torch.cuda.empty_cache()

        return processing.Processed(
            self,
            images_list=output_images,
            seed=self.all_seeds[0],
            info="",
            comments="",
            subseed=self.all_subseeds[0],
            index_of_first_image=index_of_first_image,
            infotexts=[],
        )


@dataclass(repr=False)
class ONNXStableDiffusionProcessingImg2Img(ONNXStableDiffusionProcessing):
    init_images: list = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = None
    latent_mask: Image = None

    image_mask: Any = field(default=None, init=False)

    nmask: torch.Tensor = field(default=None, init=False)
    image_conditioning: torch.Tensor = field(default=None, init=False)
    init_img_hash: str = field(default=None, init=False)
    mask_for_overlay: Image = field(default=None, init=False)
    init_latent: torch.Tensor = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        self.initial_noise_multiplier = (
            shared.opts.initial_noise_multiplier
            if self.initial_noise_multiplier is None
            else self.initial_noise_multiplier
        )

    def forward(self) -> processing.Processed:
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            # image_mask is passed in as RGBA by Gradio to support alpha masks,
            # but we still want to support binary masks.
            image_mask = processing.create_binary_mask(image_mask)

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert("L")
                crop_region = masking.get_crop_region(
                    np.array(mask), self.inpaint_full_res_padding
                )
                crop_region = masking.expand_crop_region(
                    crop_region, self.width, self.height, mask.width, mask.height
                )
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
            else:
                image_mask = images.resize_image(
                    self.resize_mode, image_mask, self.width, self.height
                )
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(
                    np.uint8
                )
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = (
            shared.opts.img2img_color_correction and self.color_corrections is None
        )
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:
            # Save init image
            if shared.opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(
                    img,
                    path=shared.opts.outdir_init_images,
                    basename=None,
                    forced_filename=self.init_img_hash,
                    save_to_dirs=False,
                )

            image = images.flatten(img, shared.opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(
                    self.resize_mode, image, self.width, self.height
                )

            if image_mask is not None:
                image_masked = Image.new("RGBa", (image.width, image.height))
                image_masked.paste(
                    image.convert("RGBA").convert("RGBa"),
                    mask=ImageOps.invert(self.mask_for_overlay.convert("L")),
                )

                self.overlay_images.append(image_masked.convert("RGBA"))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(processing.setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(
                self.batch_size, axis=0
            )
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(
                f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less"
            )

        image = 2.0 * torch.from_numpy(batch_images) - 0.7878377

        if isinstance(self.prompt, list):
            assert len(self.prompt) > 0
        else:
            assert self.prompt is not None

        gc.collect()
        torch.cuda.empty_cache()

        seed = processing.get_fixed_seed(self.seed)
        subseed = processing.get_fixed_seed(self.subseed)

        if isinstance(seed, list):
            self.all_seeds = seed
        else:
            self.all_seeds = [
                int(seed) + (x if self.subseed_strength == 0 else 0)
                for x in range(len(self.all_prompts))
            ]

        if isinstance(subseed, list):
            self.all_subseeds = subseed
        else:
            self.all_subseeds = [int(subseed) + x for x in range(len(self.all_prompts))]

        if shared.state.job_count == -1:
            shared.state.job_count = self.n_iter * self.steps

        output_images = []

        for i in range(0, self.n_iter):
            result: diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput = self.pipeline(
                self,
                image=image,
                # mask=self.image_mask,
                prompt=self.all_prompts,
                negative_prompt=self.all_negative_prompts,
                num_inference_steps=self.steps,
                num_images_per_prompt=1,
                strength=self.denoising_strength,
                eta=self.eta,
                seed=seed + i,
            )
            for image in result.images:
                images.save_image(image, self.outpath_samples, "")
            output_images += result.images

            result.images = None
            result = None
            gc.collect()
            torch.cuda.empty_cache()

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = (
            len(output_images) < 2 and shared.opts.grid_only_if_multiple
        )
        if (
            shared.opts.return_grid or shared.opts.grid_save
        ) and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, self.n_iter)

            if shared.opts.return_grid:
                output_images.insert(0, grid)
                index_of_first_image = 1

            if shared.opts.grid_save:
                images.save_image(
                    grid,
                    self.outpath_grids,
                    "grid",
                    self.all_seeds[0],
                    self.all_prompts[0],
                    shared.opts.grid_format,
                    short_filename=not shared.opts.grid_extended_filename,
                    grid=True,
                )

        gc.collect()
        torch.cuda.empty_cache()

        return processing.Processed(
            self,
            images_list=output_images,
            seed=self.all_seeds[0],
            info="",
            comments="",
            subseed=self.all_subseeds[0],
            index_of_first_image=index_of_first_image,
            infotexts=[],
        )
