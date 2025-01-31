from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spfa_csr',
    ext_modules=[
        CUDAExtension('spfa_csr', [
            'sp_flatt_csr.cpp',
            'sp_flatt_csr_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })