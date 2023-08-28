from optimum.onnxruntime import ORTStableDiffusionXLPipeline, ORTStableDiffusionXLImg2ImgPipeline

from modules import shared
from modules.sd_samplers_common import SamplerData
from modules.sd_onnx import BaseONNXModel

class ONNXStableDiffusionXLModel(BaseONNXModel[ORTStableDiffusionXLPipeline, ORTStableDiffusionXLImg2ImgPipeline]):
    def __init__(self, dirname: str, is_optimized: bool = False):
        super().__init__(dirname, is_optimized)
        self.is_sdxl = True
        self._sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        self._sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        self._sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        self._sess_options.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    def create_txt2img_pipeline(self, sampler: SamplerData) -> ORTStableDiffusionXLPipeline:
        provider_options = dict()
        provider_options["device_id"] = self.device.index
        return ORTStableDiffusionXLPipeline.from_pretrained(
            self.path,
            provider=("DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider", provider_options),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            sess_options=self._sess_options,
            local_files_only=True,
            torch_dtype=self.dtype,
            offload_state_dict=shared.opts.offload_state_dict,
        )

    def create_img2img_pipeline(self, sampler: SamplerData) -> ORTStableDiffusionXLImg2ImgPipeline:
        provider_options = dict()
        provider_options["device_id"] = self.device.index
        return ORTStableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.path,
            provider=("DmlExecutionProvider" if shared.cmd_opts.backend == "directml" else "CUDAExecutionProvider", provider_options),
            scheduler=sampler.constructor.from_pretrained(self.path, subfolder="scheduler"),
            sess_options=self._sess_options,
            local_files_only=True,
            torch_dtype=self.dtype,
            offload_state_dict=shared.opts.offload_state_dict,
        )
