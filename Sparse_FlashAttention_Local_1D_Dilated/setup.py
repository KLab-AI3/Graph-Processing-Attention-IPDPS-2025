from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_local_1d_dilated',
    ext_modules=[
        CUDAExtension('spfa_local_1d_dilated', [
            'sp_flatt_local_1d_dilated.cpp',
            'sp_flatt_local_1d_dilated_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })