from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_bsr',
    ext_modules=[
        CUDAExtension('spfa_bsr', [
            'sp_flatt_bsr.cpp',
            'sp_flatt_bsr_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })