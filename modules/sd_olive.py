import json

from modules.sd_onnx import SdONNXProcessingTxt2Img

class SdOptimizedONNXProcessingTxt2Img(SdONNXProcessingTxt2Img):
    opt_config: dict

    def __init__(self, *args, **kwargs):
        super(SdOptimizedONNXProcessingTxt2Img, self).__init__(*args, **kwargs)
        self.opt_config = json.load(open(self.sd_model.path / "opt_config.json", "r"))
        self.sd_model.sess_options.add_free_dimension_override_by_name("unet_sample_height", self.opt_config["sample_height_dim"] or 64)
        self.sd_model.sess_options.add_free_dimension_override_by_name("unet_sample_width", self.opt_config["sample_width_dim"] or 64)
