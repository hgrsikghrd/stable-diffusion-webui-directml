from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline

from modules import shared
from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel, device_map

class ONNXStableDiffusionModel(BaseONNXModel[OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline]):
    def __init__(self, dirname: str):
        super().__init__(dirname)
        self._sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        self._sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        self._sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

    def create_txt2img_pipeline(self, sampler: SamplerData) -> OnnxStableDiffusionPipeline:
        provider_options = dict()
        provider_options["device_id"] = self.device.index
        if "text_encoder" not in device_map:
            return OnnxStableDiffusionPipeline.from_pretrained(
                self.path,
                provider=("DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider", provider_options),
                scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
                sess_options=self._sess_options,
                **self.get_pipeline_config(),
            )
        return OnnxStableDiffusionPipeline(
            safety_checker=None,
            text_encoder=self.load_orm("text_encoder"),
            unet=self.load_orm("unet"),
            vae_decoder=self.load_orm("vae_decoder"),
            vae_encoder=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=self.load_image_processor("feature_extractor"),
            requires_safety_checker=False,
        )

    def create_img2img_pipeline(self, sampler: SamplerData) -> OnnxStableDiffusionImg2ImgPipeline:
        provider_options = dict()
        provider_options["device_id"] = self.device.index
        if "text_encoder" not in device_map:
            return OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                self.path,
                provider=("DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider", provider_options),
                scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
                sess_options=self._sess_options,
                **self.get_pipeline_config(),
            )
        return OnnxStableDiffusionImg2ImgPipeline(
            safety_checker=None,
            text_encoder=self.load_orm("text_encoder"),
            unet=self.load_orm("unet"),
            vae_decoder=self.load_orm("vae_decoder"),
            vae_encoder=self.load_orm("vae_encoder"),
            tokenizer=self.load_tokenizer("tokenizer"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=self.load_image_processor("feature_extractor"),
            requires_safety_checker=False,
        )
