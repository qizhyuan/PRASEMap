import os
from setuptools import setup, Extension, find_packages


def get_extension():
    file_path = os.path.abspath(__file__)
    base, _ = os.path.split(file_path)
    pybind_path = os.path.join(base, "dependence/pybind11-2.6/include")
    eigen_path = os.path.join(base, "dependence/eigen-3.3.9")
    core_path = os.path.join(base, "pr/prase_core.cpp")
    prase_core_module = Extension(name='prase_core', sources=[core_path], include_dirs=[pybind_path, eigen_path])
    return [prase_core_module]


setup(name="prase",
      version="0.1.1",
      packages=["prase", "se"],
      author="qizhyuan",
      author_email="qizhyuan@gmail.com",
      ext_modules=get_extension())
