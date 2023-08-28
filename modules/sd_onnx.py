import gc
import cv2
import torch
import numpy as np
import hashlib
import diffusers
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any
from pathlib import Path
from PIL import Image, ImageOps
import onnxruntime as ort

from modules import shared, images, devices, masking
from modules.sd_onnx_hijack import do_hijack
from modules.sd_models import reload_model_weights
from modules.sd_samplers import find_sampler_config
from modules.sd_samplers_common import SamplerData
from modules.paths_internal import models_path
from modules.processing import Processed, get_fixed_seed, setup_color_correction

do_hijack()

device_map = dict()

def save_device_map(text_encoder_ep: str, text_encoder_id: str, text_encoder_2_ep: str, text_encoder_2_id: str, unet_ep: str, unet_id: str, vae_decoder_ep: str, vae_decoder_id: str, vae_encoder_ep: str, vae_encoder_id: str):
    global device_map
    device_map["text_encoder"] = (text_encoder_ep, {
        "device_id": int(text_encoder_id)
    })
    device_map["text_encoder_2"] = (text_encoder_2_ep, {
        "device_id": int(text_encoder_2_id)
    })
    device_map["unet"] = (unet_ep, {
        "device_id": int(unet_id)
    })
    device_map["vae_decoder"] = (vae_decoder_ep, {
        "device_id": int(vae_decoder_id)
    })
    device_map["vae_encoder"] = (vae_encoder_ep, {
        "device_id": int(vae_encoder_id)
    })

    return ["Applied"]

T2I = TypeVar("T2I")
I2I = TypeVar("I2I")
class BaseONNXModel(Generic[T2I, I2I], metaclass=ABCMeta):
    _sess_options: ort.SessionOptions

    dirname: str
    is_sdxl: bool = False
    is_optimized: bool
    path: Path
    sd_model_hash: None = None
    cond_stage_model: torch.nn.Module = torch.nn.Module()
    cond_stage_key: str = ""
    vae: None = None

    dtype: torch.dtype = devices.dtype
    device: torch.device = devices.device

    def __init__(self, dirname: str, is_optimized: bool = False):
        self.dirname = dirname
        self.is_optimized = is_optimized
        self.path = Path(models_path) / ("ONNX-Olive" if self.is_optimized else "ONNX") / dirname
        self._sess_options = ort.SessionOptions()

    def to(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.dtype = arg
            if isinstance(arg, torch.device):
                self.device = arg

        for key in kwargs:
            if key == "dtype":
                self.dtype = kwargs[key]
            if key == "device":
                self.device = kwargs[key]

        return self

    def add_free_dimension_override_by_name(self, *args, **kwargs):
        return self._sess_options.add_free_dimension_override_by_name(*args, **kwargs)

    def set_mem_pattern(self, enabled: bool):
        self._sess_options.enable_mem_pattern = enabled

    def set_mem_reuse(self, enabled: bool):
        self._sess_options.enable_mem_reuse = enabled

    @abstractmethod
    def create_txt2img_pipeline(self, sampler: SamplerData) -> T2I:
        pass

    @abstractmethod
    def create_img2img_pipeline(self, sampler: SamplerData) -> I2I:
        pass

class ONNXStableDiffusionProcessing(metaclass=ABCMeta):
    sd_model: BaseONNXModel[diffusers.DiffusionPipeline, diffusers.DiffusionPipeline]
    pipeline: diffusers.DiffusionPipeline
    outpath_samples: str
    outpath_grids: str
    prompt: str
    prompt_for_display: str | None = None
    negative_prompt: str
    styles: list
    seed: int
    subseed: int
    subseed_strength: float
    seed_resize_from_h: int
    seed_resize_from_w: int
    sampler_name: str
    batch_size: int
    n_iter: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    restore_faces: bool
    tiling: bool
    do_not_save_samples: bool
    do_not_save_grid: bool
    extra_generation_params: dict

    def __init__(self, sd_model: BaseONNXModel[diffusers.DiffusionPipeline, diffusers.DiffusionPipeline]=None, outpath_samples=None, outpath_grids=None, prompt: str = "", styles: List[str] = None, seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1, seed_enable_extras: bool = True, sampler_name: str = None, batch_size: int = 1, n_iter: int = 1, steps: int = 50, cfg_scale: float = 7.0, width: int = 512, height: int = 512, restore_faces: bool = False, tiling: bool = False, do_not_save_samples: bool = False, do_not_save_grid: bool = False, extra_generation_params: Dict[Any, Any] = None, overlay_images: Any = None, negative_prompt: str = None, eta: float = None, do_not_reload_embeddings: bool = False, denoising_strength: float = 0, ddim_discretize: str = None, s_min_uncond: float = 0.0, s_churn: float = 0.0, s_tmax: float = None, s_tmin: float = 0.0, s_noise: float = 1.0, override_settings: Dict[str, Any] = None, override_settings_restore_afterwards: bool = True, sampler_index: int = None, script_args: list = None):
        self.sd_model: BaseONNXModel[diffusers.DiffusionPipeline, diffusers.DiffusionPipeline] = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_name: str = sampler_name
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = denoising_strength
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize or shared.opts.ddim_discretize
        self.s_min_uncond = s_min_uncond or shared.opts.s_min_uncond
        self.s_churn = s_churn or shared.opts.s_churn
        self.s_tmin = s_tmin or shared.opts.s_tmin
        self.s_tmax = s_tmax or float('inf')  # not representable as a standard ui option
        self.s_noise = s_noise or shared.opts.s_noise
        self.override_settings = {k: v for k, v in (override_settings or {}).items() if k not in shared.restricted_opts}
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.scripts = None
        self.script_args = script_args
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.iteration = 0
        self.is_hr_pass = False
        self.sampler = find_sampler_config(self.sampler_name)
        if self.sampler is None:
            raise Exception("Unknown sampler.")

        self.prompts = None
        self.negative_prompts = None
        self.seeds = None
        self.subseeds = None

        self.step_multiplier = 1
        self.cached_uc = [None, None]
        self.cached_c = [None, None]
        self.uc = None
        self.c = None

        if type(prompt) == list:
            self.all_prompts = self.prompt
        else:
            self.all_prompts = self.batch_size * [self.prompt]

        if type(self.negative_prompt) == list:
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = self.batch_size * [self.negative_prompt]

        self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_negative_prompts]

        self.extra_generation_params: dict = {}
        self.override_settings = {k: v for k, v in ({}).items() if k not in shared.restricted_opts}
        self.override_settings_restore_afterwards = False

        self.sd_model.set_mem_pattern(shared.opts.enable_mem_pattern)
        self.sd_model.set_mem_reuse(shared.opts.enable_mem_reuse)
        self.sd_model.add_free_dimension_override_by_name("unet_sample_batch", self.batch_size * 2)
        self.sd_model.add_free_dimension_override_by_name("unet_hidden_batch", self.batch_size * 2)
        if self.sd_model.is_sdxl:
            self.sd_model.add_free_dimension_override_by_name("unet_text_embeds_batch", self.batch_size * 2)
            self.sd_model.add_free_dimension_override_by_name("unet_text_embeds_size", self.height + 256)
            self.sd_model.add_free_dimension_override_by_name("unet_time_ids_batch", self.batch_size * 2)

    @abstractmethod
    def forward(self) -> Processed:
        pass

    def __call__(self) -> Processed:
        return self.forward()

    def close(self):
        return


class ONNXStableDiffusionProcessingTxt2Img(ONNXStableDiffusionProcessing):
    enable_hr: bool
    denoising_strength: float
    hr_scale: float
    hr_upscaler: str
    hr_second_pass_steps: int
    hr_resize_x: int
    hr_resize_y: int
    hr_upscale_to_x: int
    hr_upscale_to_y: int
    hr_sampler_name: str
    hr_prompt: str
    hr_negative_prompt: str

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, hr_sampler_name: str = None, hr_prompt: str = '', hr_negative_prompt: str = '', **kwargs):
        super().__init__(**kwargs)

        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_upscale_to_x = hr_resize_x
        self.hr_upscale_to_y = hr_resize_y
        self.hr_sampler_name = hr_sampler_name
        self.hr_prompt = hr_prompt
        self.hr_negative_prompt = hr_negative_prompt
        self.all_hr_prompts = None
        self.all_hr_negative_prompts = None

        if firstphase_width != 0 or firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = firstphase_width
            self.height = firstphase_height

        self.truncate_x = 0
        self.truncate_y = 0
        self.applied_old_hires_behavior_to = None

        self.hr_prompts = None
        self.hr_negative_prompts = None
        self.hr_extra_network_data = None

        self.cached_hr_uc = list()
        self.cached_hr_c = list()
        self.hr_c = None
        self.hr_uc = None

        if not shared.opts.reload_model_before_each_generation:
            self.pipeline = self.sd_model.create_txt2img_pipeline(self.sampler)

    def forward(self) -> Processed:
        if type(self.prompt) == list:
            assert(len(self.prompt) > 0)
        else:
            assert self.prompt is not None

        gc.collect()
        torch.cuda.empty_cache()

        seed = get_fixed_seed(self.seed)
        subseed = get_fixed_seed(self.subseed)

        if type(seed) == list:
            self.all_seeds = seed
        else:
            self.all_seeds = [int(seed) + (x if self.subseed_strength == 0 else 0) for x in range(len(self.all_prompts))]

        if type(subseed) == list:
            self.all_subseeds = subseed
        else:
            self.all_subseeds = [int(subseed) + x for x in range(len(self.all_prompts))]

        if shared.state.job_count == -1:
            shared.state.job_count = self.n_iter * self.steps

        output_images = []

        for i in range(0, self.n_iter):
            if shared.opts.reload_model_before_each_generation:
                self.sd_model = None
                self.pipeline = None
                gc.collect()
                torch.cuda.empty_cache()
                self.sd_model = reload_model_weights()
                self.pipeline = self.sd_model.create_txt2img_pipeline(self.sampler)
            result = self.pipeline(self,
                prompt=self.all_prompts,
                negative_prompt=self.all_negative_prompts,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
                #eta=self.eta,
                seed=seed + i,
            )
            image = result.images[0]
            images.save_image(image, self.outpath_samples, "")
            output_images.append(image)

            result.images = None
            result = None
            image = None
            gc.collect()
            torch.cuda.empty_cache()

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple
        if (shared.opts.return_grid or shared.opts.grid_save) and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, self.n_iter)

            if shared.opts.return_grid:
                output_images.insert(0, grid)
                index_of_first_image = 1

            if shared.opts.grid_save:
                images.save_image(grid, self.outpath_grids, "grid", self.all_seeds[0], self.all_prompts[0], shared.opts.grid_format, short_filename=not shared.opts.grid_extended_filename, grid=True)

        gc.collect()
        torch.cuda.empty_cache()

        return Processed(
            self,
            images_list=output_images,
            seed=self.all_seeds[0],
            info="",
            comments="",
            subseed=self.all_subseeds[0],
            index_of_first_image=index_of_first_image,
            infotexts=[],
        )

    def close(self):
        return


class ONNXStableDiffusionProcessingImg2Img(ONNXStableDiffusionProcessing):
    init_images: list
    resize_mode: int
    denoising_strength: float
    image_cfg_scale: float

    def __init__(self, init_images: list = None, resize_mode: int = 0, denoising_strength: float = 0.75, image_cfg_scale: float = None, mask: Any = None, mask_blur: int = None, mask_blur_x: int = 4, mask_blur_y: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = True, inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.image_cfg_scale: float = image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None
        self.init_latent = None
        self.image_mask = mask
        self.latent_mask = None
        self.mask_for_overlay = None
        if mask_blur is not None:
            mask_blur_x = mask_blur
            mask_blur_y = mask_blur
        self.mask_blur_x = mask_blur_x
        self.mask_blur_y = mask_blur_y
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.initial_noise_multiplier = shared.opts.initial_noise_multiplier if initial_noise_multiplier is None else initial_noise_multiplier
        self.mask = None
        self.nmask = None
        self.image_conditioning = None

        if not shared.opts.reload_model_before_each_generation:
            self.pipeline = self.sd_model.create_img2img_pipeline(self.sampler)

    def forward(self) -> Processed:
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            image_mask = image_mask.convert('L')

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(4 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(4 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = shared.opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:

            # Save init image
            if shared.opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(img, path=shared.opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

            image = images.flatten(img, shared.opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.

        if type(self.prompt) == list:
            assert(len(self.prompt) > 0)
        else:
            assert self.prompt is not None

        gc.collect()
        torch.cuda.empty_cache()

        seed = get_fixed_seed(self.seed)
        subseed = get_fixed_seed(self.subseed)

        if type(seed) == list:
            self.all_seeds = seed
        else:
            self.all_seeds = [int(seed) + (x if self.subseed_strength == 0 else 0) for x in range(len(self.all_prompts))]

        if type(subseed) == list:
            self.all_subseeds = subseed
        else:
            self.all_subseeds = [int(subseed) + x for x in range(len(self.all_prompts))]

        if shared.state.job_count == -1:
            shared.state.job_count = self.n_iter * self.steps

        output_images = []

        for i in range(0, self.n_iter):
            if shared.opts.reload_model_before_each_generation:
                self.sd_model = None
                self.pipeline = None
                gc.collect()
                torch.cuda.empty_cache()
                self.sd_model = reload_model_weights()
                self.pipeline = self.sd_model.create_img2img_pipeline(self.sampler)
            result = self.pipeline(self,
                image=image,
                prompt=self.all_prompts,
                negative_prompt=self.all_negative_prompts,
                num_inference_steps=self.steps,
                strength=self.denoising_strength,
                #eta=self.eta,
                seed=seed + i,
            )
            image = result.images[0]
            images.save_image(image, self.outpath_samples, "")
            output_images.append(image)

            result.images = None
            result = None
            image = None
            gc.collect()
            torch.cuda.empty_cache()

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple
        if (shared.opts.return_grid or shared.opts.grid_save) and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, self.n_iter)

            if shared.opts.return_grid:
                output_images.insert(0, grid)
                index_of_first_image = 1

            if shared.opts.grid_save:
                images.save_image(grid, self.outpath_grids, "grid", self.all_seeds[0], self.all_prompts[0], shared.opts.grid_format, short_filename=not shared.opts.grid_extended_filename, grid=True)

        gc.collect()
        torch.cuda.empty_cache()

        return Processed(
            self,
            images_list=output_images,
            seed=self.all_seeds[0],
            info="",
            comments="",
            subseed=self.all_subseeds[0],
            index_of_first_image=index_of_first_image,
            infotexts=[],
        )

    def close(self):
        return
