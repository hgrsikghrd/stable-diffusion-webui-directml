from optimum.onnxruntime import (
    ORTStableDiffusionXLPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
)

from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel


class ONNXStableDiffusionXLModel(
    BaseONNXModel[
        ORTStableDiffusionXLPipeline,
        ORTStableDiffusionXLImg2ImgPipeline,
        ORTStableDiffusionXLImg2ImgPipeline,  # optimum does not have SDXL pipeline for inpainting.
    ]
):
    is_sdxl = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sess_options.add_free_dimension_override_by_name(
            "unet_sample_channels", 4
        )
        self._sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        self._sess_options.add_free_dimension_override_by_name(
            "unet_hidden_sequence", 77
        )
        self._sess_options.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    def create_txt2img_pipeline(
        self, sampler: SamplerData
    ) -> ORTStableDiffusionXLPipeline:
        return ORTStableDiffusionXLPipeline(
            text_encoder_session=self.load_inference_session("text_encoder"),
            text_encoder_2_session=self.load_inference_session("text_encoder_2"),
            unet_session=self.load_inference_session("unet"),
            vae_decoder_session=self.load_inference_session("vae_decoder"),
            vae_encoder_session=self.load_inference_session("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            tokenizer_2=self.load_tokenizer("tokenizer_2"),
            scheduler=sampler.constructor.from_pretrained(
                self.path, subfolder="scheduler"
            ),
            feature_extractor=self.load_image_processor("feature_extractor"),
            config=self.get_pipeline_config(),
        )

    def create_img2img_pipeline(
        self, sampler: SamplerData
    ) -> ORTStableDiffusionXLImg2ImgPipeline:
        return ORTStableDiffusionXLImg2ImgPipeline(
            text_encoder_session=self.load_inference_session("text_encoder"),
            text_encoder_2_session=self.load_inference_session("text_encoder_2"),
            unet_session=self.load_inference_session("unet"),
            vae_decoder_session=self.load_inference_session("vae_decoder"),
            vae_encoder_session=self.load_inference_session("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            tokenizer_2=self.load_tokenizer("tokenizer_2"),
            scheduler=sampler.constructor.from_pretrained(
                self.path, subfolder="scheduler"
            ),
            feature_extractor=self.load_image_processor("feature_extractor"),
            config=self.get_pipeline_config(),
        )

    def create_inpaint_pipeline(
        self, sampler: SamplerData
    ) -> ORTStableDiffusionXLImg2ImgPipeline:
        print(
            "WARNING: There's no pipeline for SDXL inpainting at this time. Inpaint tab won't work as intended with SDXL models."
        )
        return self.create_img2img_pipeline(sampler)
