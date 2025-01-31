from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_global_no_local',
    ext_modules=[
        CUDAExtension('spfa_global_no_local', [
            'sp_flatt_global_no_local.cpp',
            'sp_flatt_global_no_local_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })