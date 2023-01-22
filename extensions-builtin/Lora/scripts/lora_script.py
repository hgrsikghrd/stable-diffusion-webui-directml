import torch

import lora
import extra_networks_lora
import ui_extra_networks_lora
from modules import script_callbacks, ui_extra_networks, extra_networks


def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora


def before_ui():
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())
    extra_networks.register_extra_network(extra_networks_lora.ExtraNetworkLora())


def lora_Linear_forward(self, x):
    if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
        self.weight = torch.nn.Parameter(self.weight.float())
        if self.bias is not None and self.bias.dtype == torch.float16:
            self.bias = torch.nn.Parameter(self.bias.float())
        return lora.lora_Linear_forward(x.float()).type(x.dtype)
    else:
        return lora.lora_Linear_forward(x)


def lora_Conv2d_forward(self, x):
    if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
        self.weight = torch.nn.Parameter(self.weight.float())
        if self.bias is not None and self.bias.dtype == torch.float16:
            self.bias = torch.nn.Parameter(self.bias.float())
        return lora.lora_Conv2d_forward(x.float()).type(x.dtype)
    else:
        return lora.lora_Conv2d_forward(x)


if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

torch.nn.Linear.forward = lora_Linear_forward
torch.nn.Conv2d.forward = lora_Conv2d_forward

script_callbacks.on_model_loaded(lora.assign_lora_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
