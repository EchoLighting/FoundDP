from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'dfdp/DPANet/utils/src/deform_conv_cuda.cpp',
            'dfdp/DPANet/utils/src/deform_conv_cuda_kernel.cu',
        ]),
        # CUDAExtension('deform_pool_cuda', [
        #     'dfdp/DPANet/utils/src/deform_pool_cuda.cpp', 'dfdp/DPANet/utils/src/deform_pool_cuda_kernel.cu'
        # ]),
    ],
    cmdclass={'build_ext': BuildExtension})
