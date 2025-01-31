from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_local',
    ext_modules=[
        CUDAExtension('spfa_local', [
            'sp_flatt_local.cpp',
            'sp_flatt_local_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })