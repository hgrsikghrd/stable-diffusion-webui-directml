import ctypes as C
from .atiadlxx_apis import *
from .atiadlxx_structures import *
from .atiadlxx_defines import *

import torch_directml

device = torch_directml.device(torch_directml.default_device())
device_name = torch_directml.device_name(device.index)

class ATIADLxx(object):
    iHyperMemorySize = 0

    def __init__(self):
        self.context = ADL_CONTEXT_HANDLE()
        ADL2_Main_Control_Create(ADL_Main_Memory_Alloc, 1, C.byref(self.context))
        num_adapters = C.c_int(-1)
        ADL2_Adapter_NumberOfAdapters_Get(self.context, C.byref(num_adapters))
        AdapterInfoArray = (AdapterInfo * num_adapters.value)()
        ADL2_Adapter_AdapterInfo_Get(self.context, C.cast(AdapterInfoArray, LPAdapterInfo), C.sizeof(AdapterInfoArray))
        self.devices = []
        for adapter in AdapterInfoArray:
            self.devices.append(adapter.iAdapterIndex)
        self.iHyperMemorySize = self.getMemoryInfo2(0).iHyperMemorySize

    def getMemoryInfo2(self, adapterIndex):
        info = ADLMemoryInfo2()

        if ADL2_Adapter_MemoryInfo2_Get(self.context, adapterIndex, C.byref(info)) != ADL_OK:
            raise RuntimeError("Failed to get MemoryInfo2")
        
        return info

    def getDedicatedVRAMUsage(self, adapterIndex):
        usage = C.c_int(-1)

        if ADL2_Adapter_DedicatedVRAMUsage_Get(self.context, adapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("Failed to get DedicatedVRAMUsage")

        return usage.value

    def getVRAMUsage(self, adapterIndex):
        usage = C.c_int(-1)

        if ADL2_Adapter_VRAMUsage_Get(self.context, adapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("Failed to get VRAMUsage")

        return usage.value

    def memory_stats(self):
        return (self.iHyperMemorySize, self.getDedicatedVRAMUsage(0))

    def create():
        match ATIADLxx.test():
            case 0:
                return ATIADLxx()
            case 1:
                print("Warning: experimental graphic memory optimization is disabled because failed to get dedicated vram usage.")
                return None
            case 2:
                print("Warning: experimental graphic memory optimization is disabled due to gpu vendor. Currently this optimization is only available for AMDGPUs.")
                return None
            case 3:
                print("Warning: experimental graphic memory optimization for AMDGPU is disabled. Because there is an unknown error.")
                return None

    def test():
        try:
            if device.type == "privateuseone" and ("AMD" in device_name or "Radeon" in device_name):
                try:
                    ATIADLxx()
                    return 0 # OK
                except AttributeError:
                    return 1 # failed to get dedicated vram usage
            else:
                return 2 # not AMDGPU
        except RuntimeError:
            return 3 # unknown