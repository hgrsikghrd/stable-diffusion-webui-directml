import json
from os.path import isfile

from modules.sd_onnx import SdONNXProcessingTxt2Img

class SdOptimizedONNXProcessingTxt2Img(SdONNXProcessingTxt2Img):
    def __init__(self, *args, **kwargs):
        super(SdOptimizedONNXProcessingTxt2Img, self).__init__(*args, **kwargs)
        opt_config_path = self.sd_model.path / "opt_config.json"
        if isfile(opt_config_path):
            with open(opt_config_path, "r") as raw:
                opt_config = json.load(raw)
                #self.sess_options.add_free_dimension_override_by_name("unet_sample_height", opt_config["sample_height_dim"] or 64)
                #self.sess_options.add_free_dimension_override_by_name("unet_sample_width", opt_config["sample_width_dim"] or 64)
        else:
            print("Warning: Failed to assume the sample dimension. There is no opt_config.json in the model path.")
