from diffusers import OnnxStableDiffusionPipeline, DiffusionPipeline
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

tries = [
    lambda id, **kwargs: OnnxStableDiffusionPipeline.from_pretrained(id, **kwargs),
    lambda id, **kwargs: OnnxStableDiffusionPipeline.from_pretrained(id, provider="DmlExecutionProvider", **kwargs),
    lambda id, **kwargs: DiffusionPipeline.from_pretrained(id, revision="onnx", **kwargs),
    lambda id, **kwargs: DiffusionPipeline.from_pretrained(id, **kwargs),
]

tries_sdxl = [
    lambda id, **kwargs: ORTStableDiffusionXLPipeline.from_pretrained(id, **kwargs),
    lambda id, **kwargs: ORTStableDiffusionXLPipeline.from_pretrained(id, provider="DmlExecutionProvider", **kwargs),
    lambda id, **kwargs: DiffusionPipeline.from_pretrained(id, revision="onnx", **kwargs),
    lambda id, **kwargs: DiffusionPipeline.from_pretrained(id, **kwargs),
]

def load_pipeline(id: str, is_sdxl: bool = False, **kwargs):
    for load in (tries_sdxl if is_sdxl else tries):
        try:
            return load(id, **kwargs)
        except Exception:
            continue
    return None
