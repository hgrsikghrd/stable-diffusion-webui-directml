from diffusers import (
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)

from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel


class ONNXStableDiffusionModel(
    BaseONNXModel[
        OnnxStableDiffusionPipeline,
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
    ]
):
    is_sdxl = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sess_options.add_free_dimension_override_by_name(
            "unet_sample_channels", 4
        )
        self._sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        self._sess_options.add_free_dimension_override_by_name(
            "unet_hidden_sequence", 77
        )

    def create_txt2img_pipeline(
        self, sampler: SamplerData
    ) -> OnnxStableDiffusionPipeline:
        return OnnxStableDiffusionPipeline(
            safety_checker=None,
            text_encoder=self.load_orm("text_encoder"),
            unet=self.load_orm("unet"),
            vae_decoder=self.load_orm("vae_decoder"),
            vae_encoder=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            scheduler=sampler.constructor.from_pretrained(
                self.path, subfolder="scheduler"
            ),
            feature_extractor=self.load_image_processor("feature_extractor"),
            requires_safety_checker=False,
        )

    def create_img2img_pipeline(
        self, sampler: SamplerData
    ) -> OnnxStableDiffusionImg2ImgPipeline:
        return OnnxStableDiffusionImg2ImgPipeline(
            safety_checker=None,
            text_encoder=self.load_orm("text_encoder"),
            unet=self.load_orm("unet"),
            vae_decoder=self.load_orm("vae_decoder"),
            vae_encoder=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            scheduler=sampler.constructor.from_pretrained(
                self.path, subfolder="scheduler"
            ),
            feature_extractor=self.load_image_processor("feature_extractor"),
            requires_safety_checker=False,
        )

    def create_inpaint_pipeline(
        self, sampler: SamplerData
    ) -> OnnxStableDiffusionInpaintPipeline:
        print(
            "WARNING: Inpaint for Onnx models is under development. Inpaint tab won't work as intended."
        )
        return self.create_img2img_pipeline(sampler)
        return OnnxStableDiffusionInpaintPipeline(
            safety_checker=None,
            text_encoder=self.load_orm("text_encoder"),
            unet=self.load_orm("unet"),
            vae_decoder=self.load_orm("vae_decoder"),
            vae_encoder=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            scheduler=sampler.constructor.from_pretrained(
                self.path, subfolder="scheduler"
            ),
            feature_extractor=self.load_image_processor("feature_extractor"),
            requires_safety_checker=False,
        )
