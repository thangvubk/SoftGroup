from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='softgroup',
        version='1.0',
        description='SoftGroup: SoftGroup for 3D Instance Segmentation [CVPR 2022]',
        author='Thang Vu',
        author_email='thangvubk@kaist.ac.kr',
        packages=['softgroup'],
        package_data={'softgroup.ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='softgroup.ops.ops',
                sources=[
                    'softgroup/ops/src/softgroup_api.cpp', 'softgroup/ops/src/softgroup_ops.cpp',
                    'softgroup/ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                })
        ],
        cmdclass={'build_ext': BuildExtension})
