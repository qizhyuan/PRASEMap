from setuptools import setup, Extension

functions_module = Extension(
    name ='prase_core',
    sources = ['prase_core.cpp'],
    include_dirs = [r'D:\repos\study\repos\pybind11\include',
                    r"D:\repos\study\repos\eigen-3.3.9",
                   r'C:\Program Files\Python\Python37\include']
# include_dirs 中的路径 更改为自己虚拟环境下相应 pybind11的路径 和 python的include路径
)

setup(ext_modules = [functions_module])