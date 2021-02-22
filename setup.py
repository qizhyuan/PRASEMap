from setuptools import setup, Extension

functions_module = Extension(
    name ='prase',
    sources = ['prase.cpp'],
    include_dirs = [r'D:\repos\pybind11-2.6\include',
                   r'C:\ProgramData\Anaconda3\include']
# include_dirs 中的路径 更改为自己虚拟环境下相应 pybind11的路径 和 python的include路径
)

setup(ext_modules = [functions_module])