import diffusers
from modules.sd_samplers_common import SamplerData

samplers = [
    SamplerData("pndm", diffusers.PNDMScheduler, [], None),
    SamplerData("lms", diffusers.LMSDiscreteScheduler, [], None),
    SamplerData("heun", diffusers.HeunDiscreteScheduler, [], None),
    SamplerData("ddim", diffusers.DDIMScheduler, [], None),
    SamplerData("ddpm", diffusers.DDPMScheduler, [], None),
    SamplerData("euler", diffusers.EulerDiscreteScheduler, [], None),
    SamplerData("euler-ancestral", diffusers.EulerAncestralDiscreteScheduler, [], None),
    SamplerData("dpm", diffusers.DPMSolverMultistepScheduler, [], None),
]
