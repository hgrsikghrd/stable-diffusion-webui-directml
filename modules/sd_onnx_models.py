from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxRuntimeModel
from transformers import CLIPTokenizer, CLIPImageProcessor

from modules import shared
from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel, device_map

class ONNXStableDiffusionModel(BaseONNXModel[OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline]):
    def __init__(self, dirname: str, is_optimized: bool = False):
        super().__init__(dirname, is_optimized)
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
                local_files_only=True,
                torch_dtype=self.dtype,
                offload_state_dict=shared.opts.offload_state_dict,
            )
        return OnnxStableDiffusionPipeline(
            safety_checker=None,
            text_encoder=OnnxRuntimeModel.from_pretrained(self.path / "text_encoder", provider=device_map["text_encoder"]),
            unet=OnnxRuntimeModel.from_pretrained(self.path / "unet", provider=device_map["unet"]),
            vae_decoder=OnnxRuntimeModel.from_pretrained(self.path / "vae_decoder", provider=device_map["vae_decoder"]),
            vae_encoder=OnnxRuntimeModel.from_pretrained(self.path / "vae_encoder", provider=device_map["vae_encoder"]),
            tokenizer=CLIPTokenizer.from_pretrained(self.path / "tokenizer"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=CLIPImageProcessor.from_pretrained(self.path / "feature_extractor"),
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
                local_files_only=True,
                torch_dtype=self.dtype,
                offload_state_dict=shared.opts.offload_state_dict,
            )
        return OnnxStableDiffusionImg2ImgPipeline(
            safety_checker=None,
            text_encoder=OnnxRuntimeModel.from_pretrained(self.path / "text_encoder", provider=device_map["text_encoder"]),
            unet=OnnxRuntimeModel.from_pretrained(self.path / "unet", provider=device_map["unet"]),
            vae_decoder=OnnxRuntimeModel.from_pretrained(self.path / "vae_decoder", provider=device_map["vae_decoder"]),
            vae_encoder=OnnxRuntimeModel.from_pretrained(self.path / "vae_encoder", provider=device_map["vae_encoder"]),
            tokenizer=CLIPTokenizer.from_pretrained(self.path / "tokenizer"),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            feature_extractor=CLIPImageProcessor.from_pretrained(self.path / "feature_extractor"),
            requires_safety_checker=False,
        )
