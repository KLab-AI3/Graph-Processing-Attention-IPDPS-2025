from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_coo',
    ext_modules=[
        CUDAExtension('spfa_coo', [
            'sp_flatt_coo.cpp',
            'sp_flatt_coo_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })