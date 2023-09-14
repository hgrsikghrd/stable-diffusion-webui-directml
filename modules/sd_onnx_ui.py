from pathlib import Path

from modules.paths_internal import models_path
from modules.sd_onnx import device_map
from modules.sd_onnx_utils import load_pipeline

def download_from_huggingface(id: str, dest: str):
    path = Path(models_path) / dest / id.split("/")[1]
    pipeline = load_pipeline(id)
    if pipeline is None:
        pipeline = load_pipeline(id, True)

    if pipeline is None:
        return ["Failed to download model"]

    pipeline.save_pretrained(path)

    del pipeline

    return [f"Model saved: {path}"]

def save_device_map(text_encoder_ep: str, text_encoder_id: str, text_encoder_2_ep: str, text_encoder_2_id: str, unet_ep: str, unet_id: str, vae_decoder_ep: str, vae_decoder_id: str, vae_encoder_ep: str, vae_encoder_id: str):
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
