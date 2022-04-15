from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SOFTGROUP_OP',
    ext_modules=[
        CUDAExtension(
            'SOFTGROUP_OP', ['src/softgroup_api.cpp', 'src/softgroup_ops.cpp', 'src/cuda.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            })
    ],
    cmdclass={'build_ext': BuildExtension})
