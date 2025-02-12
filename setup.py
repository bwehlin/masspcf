from skbuild import setup
from os import environ

name_suffix='-cpu'

if environ.get('BUILD_WITH_CUDA') is not None and environ.get('BUILD_WITH_CUDA') != '0':
    name_suffix=''

setup(
    name=f'masspcf{name_suffix}',
    version='0.3.2',
    author='Bjorn H. Wehlin',
    license='TBD',
    packages=['masspcf'],
    python_requires=">=3.10",
    cmake_args=['-DCMAKE_BUILD_TYPE=Release']
)
