from optimum.onnxruntime import ORTStableDiffusionXLPipeline, ORTStableDiffusionXLImg2ImgPipeline

from modules import shared
from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel, device_map

class ONNXStableDiffusionXLModel(BaseONNXModel[ORTStableDiffusionXLPipeline, ORTStableDiffusionXLImg2ImgPipeline]):
    is_sdxl = True

    def __init__(self, dirname: str):
        super().__init__(dirname)
        self._sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        self._sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        self._sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        self._sess_options.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    def create_txt2img_pipeline(self, sampler: SamplerData) -> ORTStableDiffusionXLPipeline:
        if "text_encoder" not in device_map:
            return ORTStableDiffusionXLPipeline.from_pretrained(
                self.path,
                provider="DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider",
                scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
                sess_options=self._sess_options,
                **self.get_pipeline_config(),
            )
        return ORTStableDiffusionXLPipeline(
            text_encoder_session=self.load_orm("text_encoder"),
            text_encoder_2_session=self.load_orm("text_encoder_2"),
            unet_session=self.load_orm("unet"),
            vae_decoder_session=self.load_orm("vae_decoder"),
            vae_encoder_session=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            tokenizer_2=self.load_tokenizer("tokenizer_2"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=self.load_image_processor("feature_extractor"),
            config=self.get_pipeline_config(),
        )

    def create_img2img_pipeline(self, sampler: SamplerData) -> ORTStableDiffusionXLImg2ImgPipeline:
        if "text_encoder" not in device_map:
            return ORTStableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.path,
                provider="DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider",
                scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
                sess_options=self._sess_options,
                **self.get_pipeline_config(),
            )
        return ORTStableDiffusionXLImg2ImgPipeline(
            text_encoder_session=self.load_orm("text_encoder"),
            text_encoder_2_session=self.load_orm("text_encoder_2"),
            unet_session=self.load_orm("unet"),
            vae_decoder_session=self.load_orm("vae_decoder"),
            vae_encoder_session=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            tokenizer_2=self.load_tokenizer("tokenizer_2"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=self.load_image_processor("feature_extractor"),
            config=self.get_pipeline_config(),
        )
