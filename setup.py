from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["akf_lib/regressor.pyx", "akf_lib/kernels.pyx"]), include_path='./akf_lib'
)