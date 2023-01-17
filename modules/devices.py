import sys, os, shlex
import contextlib
import torch
import torch_directml
from modules import errors
from packaging import version
from functools import reduce
import operator


# has_mps is only available in nightly pytorch (for now) and macOS 12.3+.
# check `getattr` and try it for compatibility
def has_mps() -> bool:
    if not getattr(torch, 'has_mps', False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_cuda_device_string():
    from modules import shared

    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device(get_cuda_device_string())

    if has_mps():
        return torch.device("mps")

    if torch_directml.is_available():
        return torch_directml.device(torch_directml.default_device())
    
    return cpu


def get_device_for(task):
    from modules import shared

    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16


def randn(seed, shape):
    torch.manual_seed(seed)
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


class NansException(Exception):
    pass


def test_for_nans(x, where):
    from modules import shared

    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try using --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."

    raise NansException(message)


# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
orig_tensor_to = torch.Tensor.to
def tensor_to_fix(self, *args, **kwargs):
    if self.device.type != 'mps' and \
       ((len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == 'mps') or \
       (isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps')):
        self = self.contiguous()
    return orig_tensor_to(self, *args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/80800 
orig_layer_norm = torch.nn.functional.layer_norm
def layer_norm_fix(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps':
        args = list(args)
        args[0] = args[0].contiguous()
    return orig_layer_norm(*args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/90532
orig_tensor_numpy = torch.Tensor.numpy
def numpy_fix(self, *args, **kwargs):
    if self.requires_grad:
        self = self.detach()
    return orig_tensor_numpy(self, *args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/89784
orig_cumsum = torch.cumsum
orig_Tensor_cumsum = torch.Tensor.cumsum
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if any(output_dtype == broken_dtype for broken_dtype in [torch.bool, torch.int8, torch.int16, torch.int64]):
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
    return cumsum_func(input, *args, **kwargs)


class GroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        D = int(C / G)
        NxG = N * G
        HxW = reduce(operator.mul, x.shape[2:], 1)

        x = x.view(N, G, -1)
        x = ((x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + self.eps).sqrt()).view(NxG, D, HxW)
        x = self.weight.repeat(N).view(NxG, D, 1).repeat(1, 1, HxW) * x + self.bias.repeat(N).view(NxG, D, 1).repeat(1, 1, HxW)

        x = x.view(N, C, H, W)
        return x


class _GroupNorm(GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, x):
        if x.device.type == 'privateuseone':
            dims = [-(i + 1) for i in range(len(self.normalized_shape))]
            x = (x - x.mean(dim=dims, keepdim=True)) / (x.var(dim=dims, keepdim=True) + self.eps).sqrt()
            if self.elementwise_affine:
                x = self.weight * x + self.bias
            return x
        else:
            return super().forward(x)


class Linear(torch.nn.Linear):
    def forward(self, x):
        self.weight = torch.nn.Parameter(self.weight.type(x.dtype))
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.type(x.dtype))
        return super().forward(x)


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        self.weight = torch.nn.Parameter(self.weight.type(x.dtype))
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.type(x.dtype))
        return super().forward(x)


_new_zeros = torch.Tensor.new_zeros
def new_zeros(self, *arg, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    if self.dtype == torch.float16 and self.device.type == 'privateuseone':
        return torch.zeros(*arg, requires_grad=requires_grad, layout=layout, pin_memory=pin_memory, dtype=dtype).to(self.device)
    else:
        return _new_zeros(self, *arg)


_new_ones = torch.Tensor.new_ones
def new_ones(self, *arg, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    if self.dtype == torch.float16 and self.device.type == 'privateuseone':
        return torch.ones(*arg, requires_grad=requires_grad, layout=layout, pin_memory=pin_memory, dtype=dtype).to(self.device)
    else:
        return _new_ones(self, *arg)


_var = torch.Tensor.var
def var(self, *arg, **kwarg):
    if self.dtype == torch.float16 and self.device.type == 'privateuseone':
        return _var(self.type(torch.float32), *arg, **kwarg).type(self.dtype)
    else:
        return _var(self, *arg, **kwarg)


_pow = torch.Tensor.pow
def pow(self, *arg, **kwarg):
    if self.dtype == torch.float16 and self.device.type == 'privateuseone':
        return _pow(self.type(torch.float32), *arg, **kwarg).type(self.dtype)
    else:
        return _pow(self, *arg, **kwarg)


_zeros_like = torch.zeros_like
def zeros_like(input, *args, **kwarg):
    if input.dtype == torch.float16 and input.device.type == 'privateuseone':
        return torch.zeros(input.size(), dtype=input.dtype).to(input.device)
    else:
        return _zeros_like(input, *args, **kwarg)


_cat = torch.cat
def cat(tensors, *arg, **kwarg):
    return _cat(tuple(map(lambda tensor: tensor.type(torch.float32) if tensor.dtype == torch.float16 and tensor.device.type == 'privateuseone' else tensor, tensors)), *arg, **kwarg)


if has_mps():
    if version.parse(torch.__version__) < version.parse("1.13"):
        # PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working
        torch.Tensor.to = tensor_to_fix
        torch.nn.functional.layer_norm = layer_norm_fix
        torch.Tensor.numpy = numpy_fix
    elif version.parse(torch.__version__) > version.parse("1.13.1"):
        if not torch.Tensor([1,2]).to(torch.device("mps")).equal(torch.Tensor([1,1]).to(torch.device("mps")).cumsum(0, dtype=torch.int16)):
            torch.cumsum = lambda input, *args, **kwargs: ( cumsum_fix(input, orig_cumsum, *args, **kwargs) )
            torch.Tensor.cumsum = lambda self, *args, **kwargs: ( cumsum_fix(self, orig_Tensor_cumsum, *args, **kwargs) )
        orig_narrow = torch.narrow
        torch.narrow = lambda *args, **kwargs: ( orig_narrow(*args, **kwargs).clone() )


if get_optimal_device().type == 'privateuseone':
    torch.zeros_like = zeros_like
    torch.cat = cat
    torch.Tensor.new_zeros = new_zeros
    torch.Tensor.new_ones = new_ones
    torch.Tensor.var = var
    torch.Tensor.pow = pow
    torch.Tensor.__pow__ = pow

    torch.nn.GroupNorm = GroupNorm
    torch.nn.LayerNorm = LayerNorm
    torch.nn.Linear = Linear
    torch.nn.Conv2d = Conv2d

