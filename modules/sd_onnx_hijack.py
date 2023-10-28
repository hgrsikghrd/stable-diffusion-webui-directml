import torch
import inspect
import numpy as np
from typing import Union, Callable, Optional, Tuple, Dict, List, Any
from PIL import Image
import diffusers
import optimum.pipelines.diffusers.pipeline_stable_diffusion_xl
import optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img
from optimum.pipelines.diffusers.pipeline_utils import rescale_noise_cfg
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_img2img import (
    preprocess,
)
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE

from modules import shared


def OnnxStableDiffusionPipeline__call__(
    self: diffusers.OnnxStableDiffusionPipeline,
    p,
    prompt=None,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="pil",
    return_dict: bool = True,
    callback=None,
    callback_steps: int = 1,
    seed: int = -1,
    **_,
):
    # check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if generator is None:
        generator = np.random

    if seed != -1:
        generator.seed(int(seed))

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = self._encode_prompt(
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # get the initial random noise unless the user supplied it
    latents_dtype = prompt_embeds.dtype
    latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
    if latents is None:
        latents = generator.randn(*latents_shape).astype(latents_dtype)
    elif latents.shape != latents_shape:
        raise ValueError(
            f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
        )

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    latents = latents * np.float64(self.scheduler.init_noise_sigma)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    timestep_dtype = next(
        (
            input.type
            for input in self.unet.model.get_inputs()
            if input.name == "timestep"
        ),
        "tensor(float)",
    )
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        if shared.state.skipped:
            shared.state.skipped = False

        if shared.state.interrupted:
            break

        if p.n_iter > 1:
            shared.state.job = f"Batch {i+1} out of {p.n_iter}"

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(
            torch.from_numpy(latent_model_input), t
        )
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
        )
        noise_pred = noise_pred[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = self.scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            **extra_step_kwargs,
        )
        latents = scheduler_output.prev_sample.numpy()

        # call the callback, if provided
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

        shared.state.nextjob()

    latents = 1 / 0.18215 * latents
    # image = self.vae_decoder(latent_sample=latents)[0]
    # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
    image = np.concatenate(
        [
            self.vae_decoder(latent_sample=latents[i : i + 1])[0]
            for i in range(latents.shape[0])
        ]
    )

    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))

    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="np"
        ).pixel_values.astype(image.dtype)

        images, has_nsfw_concept = [], []
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(
                clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
            )
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    else:
        has_nsfw_concept = None

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


def OnnxStableDiffusionImg2ImgPipeline__call__(
    self: diffusers.OnnxStableDiffusionImg2ImgPipeline,
    p,
    prompt: Union[str, List[str]],
    image: Union[np.ndarray, Image.Image] = None,
    strength: float = 0.8,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: Optional[float] = 0.0,
    generator: Optional[np.random.RandomState] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
    callback_steps: int = 1,
    seed: int = -1,
    **_,
):
    # check inputs. Raise error if not correct
    self.check_inputs(
        prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    # define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if strength < 0 or strength > 1:
        raise ValueError(
            f"The value of strength should in [0.0, 1.0] but is {strength}"
        )

    if generator is None:
        generator = np.random

    if seed != -1:
        generator.seed(int(seed))

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    image = preprocess(image).cpu().numpy()

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = self._encode_prompt(
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    latents_dtype = prompt_embeds.dtype
    image = image.astype(latents_dtype)
    # encode the init image into latents and scale the latents
    init_latents = self.vae_encoder(sample=image)[0]
    init_latents = 0.18215 * init_latents

    if isinstance(prompt, str):
        prompt = [prompt]
    if len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] == 0:
        additional_image_per_prompt = len(prompt) // init_latents.shape[0]
        init_latents = np.concatenate(
            [init_latents] * additional_image_per_prompt * num_images_per_prompt, axis=0
        )
    elif (
        len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] != 0
    ):
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {len(prompt)} text prompts."
        )
    else:
        init_latents = np.concatenate([init_latents] * num_images_per_prompt, axis=0)

    # get the original timestep using init_timestep
    offset = self.scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
    timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)

    # add noise to latents using the timesteps
    noise = generator.randn(*init_latents.shape).astype(latents_dtype)
    init_latents = self.scheduler.add_noise(
        torch.from_numpy(init_latents),
        torch.from_numpy(noise),
        torch.from_numpy(timesteps),
    )
    init_latents = init_latents.numpy()

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    latents = init_latents

    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = self.scheduler.timesteps[t_start:].numpy()

    timestep_dtype = next(
        (
            input.type
            for input in self.unet.model.get_inputs()
            if input.name == "timestep"
        ),
        "tensor(float)",
    )
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

    for i, t in enumerate(self.progress_bar(timesteps)):
        if shared.state.skipped:
            shared.state.skipped = False

        if shared.state.interrupted:
            break

        if p.n_iter > 1:
            shared.state.job = f"Batch {i+1} out of {p.n_iter}"

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(
            torch.from_numpy(latent_model_input), t
        )
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = self.scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            **extra_step_kwargs,
        )
        latents = scheduler_output.prev_sample.numpy()

        # call the callback, if provided
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

        shared.state.nextjob()

    latents = 1 / 0.18215 * latents
    # image = self.vae_decoder(latent_sample=latents)[0]
    # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
    image = np.concatenate(
        [
            self.vae_decoder(latent_sample=latents[i : i + 1])[0]
            for i in range(latents.shape[0])
        ]
    )

    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))

    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="np"
        ).pixel_values.astype(image.dtype)
        # safety_checker does not support batched inputs yet
        images, has_nsfw_concept = [], []
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(
                clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
            )
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    else:
        has_nsfw_concept = None

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


@torch.no_grad()
def OnnxStableDiffusionInpaintPipeline__call__(
    self: diffusers.OnnxStableDiffusionInpaintPipeline,
    p,
    prompt: Union[str, List[str]],
    image: Image.Image,
    mask: Image.Image,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[np.random.RandomState] = None,
    latents: Optional[np.ndarray] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
    callback_steps: int = 1,
    seed: int = -1,
    **_,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        image (`PIL.Image.Image`):
            `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
            be masked out with `mask_image` and repainted according to `prompt`.
        mask_image (`PIL.Image.Image`):
            `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
            repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
            to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
            instead of 3, so the expected shape would be `(B, H, W, 1)`.
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
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`np.random.RandomState`, *optional*):
            A np.random.RandomState to make generation deterministic.
        latents (`np.ndarray`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`np.ndarray`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`np.ndarray`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: np.ndarray)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """

    # check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if generator is None:
        generator = np.random

    if seed != -1:
        generator.seed(int(seed))

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = self._encode_prompt(
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    num_channels_latents = (
        diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.NUM_LATENT_CHANNELS
    )
    latents_shape = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height // 8,
        width // 8,
    )
    latents_dtype = prompt_embeds.dtype
    if latents is None:
        latents = generator.randn(*latents_shape).astype(latents_dtype)
    else:
        if latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )

    # prepare mask and masked_image
    (
        mask,
        masked_image,
    ) = diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.prepare_mask_and_masked_image(
        image, mask, latents_shape[-2:]
    )

    mask = mask.astype(latents.dtype)
    masked_image = masked_image.astype(latents.dtype)

    masked_image_latents = self.vae_encoder(sample=masked_image)[0]
    masked_image_latents = 0.18215 * masked_image_latents

    # duplicate mask and masked_image_latents for each generation per prompt
    mask = mask.repeat(batch_size * num_images_per_prompt, 0)
    masked_image_latents = masked_image_latents.repeat(
        batch_size * num_images_per_prompt, 0
    )

    mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        np.concatenate([masked_image_latents] * 2)
        if do_classifier_free_guidance
        else masked_image_latents
    )

    num_channels_mask = mask.shape[1]
    num_channels_masked_image = masked_image_latents.shape[1]

    unet_input_channels = (
        diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.NUM_UNET_INPUT_CHANNELS
    )
    if (
        num_channels_latents + num_channels_mask + num_channels_masked_image
        != unet_input_channels
    ):
        raise ValueError(
            "Incorrect configuration settings! The config of `pipeline.unet` expects"
            f" {unet_input_channels} but received `num_channels_latents`: {num_channels_latents} +"
            f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
            f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
            " `pipeline.unet` or your `mask_image` or `image` input."
        )

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * np.float64(self.scheduler.init_noise_sigma)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    timestep_dtype = next(
        (
            input.type
            for input in self.unet.model.get_inputs()
            if input.name == "timestep"
        ),
        "tensor(float)",
    )
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        if shared.state.skipped:
            shared.state.skipped = False

        if shared.state.interrupted:
            break

        if p.n_iter > 1:
            shared.state.job = f"Batch {i+1} out of {p.n_iter}"

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        )
        # concat latents, mask, masked_image_latnets in the channel dimension
        latent_model_input = self.scheduler.scale_model_input(
            torch.from_numpy(latent_model_input), t
        )
        latent_model_input = latent_model_input.cpu().numpy()
        latent_model_input = np.concatenate(
            [latent_model_input, mask, masked_image_latents], axis=1
        )

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = self.scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            **extra_step_kwargs,
        )
        latents = scheduler_output.prev_sample.numpy()

        # call the callback, if provided
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

        shared.state.nextjob()

    latents = 1 / 0.18215 * latents
    # image = self.vae_decoder(latent_sample=latents)[0]
    # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
    image = np.concatenate(
        [
            self.vae_decoder(latent_sample=latents[i : i + 1])[0]
            for i in range(latents.shape[0])
        ]
    )

    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))

    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="np"
        ).pixel_values.astype(image.dtype)
        # safety_checker does not support batched inputs yet
        images, has_nsfw_concept = [], []
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(
                clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
            )
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    else:
        has_nsfw_concept = None

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


def StableDiffusionXLPipelineMixin__call__(
    self: optimum.pipelines.diffusers.pipeline_stable_diffusion_xl.StableDiffusionXLPipelineMixin,
    p,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[np.random.RandomState] = None,
    latents: Optional[np.ndarray] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    pooled_prompt_embeds: Optional[np.ndarray] = None,
    negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    seed: int = -1,
    **_,
):
    # 0. Default height and width to unet
    height = height or self.unet.config["sample_size"] * self.vae_scale_factor
    width = width or self.unet.config["sample_size"] * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )

    # 2. Define call parameters
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if generator is None:
        generator = np.random

    if seed != -1:
        generator.seed(int(seed))

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self._encode_prompt(
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        self.unet.config.get("in_channels", 4),
        height,
        width,
        prompt_embeds.dtype,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = (original_size + crops_coords_top_left + target_size,)
    add_time_ids = np.array(add_time_ids, dtype=prompt_embeds.dtype)

    if do_classifier_free_guidance:
        prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
        add_text_embeds = np.concatenate(
            (negative_pooled_prompt_embeds, add_text_embeds), axis=0
        )
        add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
    add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

    # Adapted from diffusers to extend it for other runtimes than ORT
    timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    for i, t in enumerate(self.progress_bar(timesteps)):
        if shared.state.skipped:
            shared.state.skipped = False

        if shared.state.interrupted:
            break

        if p.n_iter > 1:
            shared.state.job = f"Batch {i+1} out of {p.n_iter}"

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(
            torch.from_numpy(latent_model_input), t
        )
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            text_embeds=add_text_embeds,
            time_ids=add_time_ids,
        )
        noise_pred = noise_pred[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = self.scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            **extra_step_kwargs,
        )
        latents = scheduler_output.prev_sample.numpy()

        # call the callback, if provided
        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        ):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        shared.state.nextjob()

    if output_type == "latent":
        image = latents
    else:
        latents = latents / self.vae_decoder.config.get("scaling_factor", 0.18215)
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )
        image = self.watermark.apply_watermark(image)

        # TODO: add image_processor
        image = np.clip(image / 2 + 0.5, 0, 1).transpose((0, 2, 3, 1))

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)


def StableDiffusionXLImg2ImgPipelineMixin__call__(
    self: optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipelineMixin,
    p,
    prompt: Optional[Union[str, List[str]]] = None,
    image: Union[np.ndarray, Image.Image] = None,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[np.random.RandomState] = None,
    latents: Optional[np.ndarray] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    pooled_prompt_embeds: Optional[np.ndarray] = None,
    negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    aesthetic_score: float = 6.0,
    negative_aesthetic_score: float = 2.5,
    seed: int = -1,
    **_,
):
    # 0. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        strength,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 1. Define call parameters
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if generator is None:
        generator = np.random

    if seed != -1:
        generator.seed(int(seed))

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 2. Encode input prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self._encode_prompt(
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    # 3. Preprocess image
    image = preprocess(image)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
    latent_timestep = np.repeat(
        timesteps[:1], batch_size * num_images_per_prompt, axis=0
    )
    timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

    latents_dtype = prompt_embeds.dtype
    image = image.astype(latents_dtype)

    # 5. Prepare latent variables
    latents = self.prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        latents_dtype,
        generator,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = {}
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    height, width = latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 8. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids, add_neg_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        dtype=prompt_embeds.dtype,
    )

    if do_classifier_free_guidance:
        prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
        add_text_embeds = np.concatenate(
            (negative_pooled_prompt_embeds, add_text_embeds), axis=0
        )
        add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
    add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    for i, t in enumerate(self.progress_bar(timesteps)):
        if shared.state.skipped:
            shared.state.skipped = False

        if shared.state.interrupted:
            break

        if p.n_iter > 1:
            shared.state.job = f"Batch {i+1} out of {p.n_iter}"

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(
            torch.from_numpy(latent_model_input), t
        )
        latent_model_input = latent_model_input.cpu().numpy()

        # predict the noise residual
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            text_embeds=add_text_embeds,
            time_ids=add_time_ids,
        )
        noise_pred = noise_pred[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = self.scheduler.step(
            torch.from_numpy(noise_pred),
            t,
            torch.from_numpy(latents),
            **extra_step_kwargs,
        )
        latents = scheduler_output.prev_sample.numpy()

        # call the callback, if provided
        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        ):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        shared.state.nextjob()

    if output_type == "latent":
        image = latents
    else:
        latents = latents / self.vae_decoder.config.get("scaling_factor", 0.18215)
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [
                self.vae_decoder(latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )
        image = self.watermark.apply_watermark(image)

        # TODO: add image_processor
        image = np.clip(image / 2 + 0.5, 0, 1).transpose((0, 2, 3, 1))

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)


def do_hijack():
    diffusers.OnnxStableDiffusionPipeline.__call__ = OnnxStableDiffusionPipeline__call__
    diffusers.OnnxStableDiffusionImg2ImgPipeline.__call__ = (
        OnnxStableDiffusionImg2ImgPipeline__call__
    )
    diffusers.OnnxStableDiffusionInpaintPipeline.__call__ = (
        OnnxStableDiffusionInpaintPipeline__call__
    )

    optimum.pipelines.diffusers.pipeline_stable_diffusion_xl.StableDiffusionXLPipelineMixin.__call__ = (
        StableDiffusionXLPipelineMixin__call__
    )
    optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipelineMixin.__call__ = (
        StableDiffusionXLImg2ImgPipelineMixin__call__
    )
