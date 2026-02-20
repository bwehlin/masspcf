from skbuild import setup
from os import environ
import tomli

name_suffix='-cpu'

version = tomli.load(open('${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml', 'rb'))['project']['version']

if environ.get('BUILD_WITH_CUDA') is not None and environ.get('BUILD_WITH_CUDA') != '0':
    name_suffix=''

setup(
    name=f'masspcf{name_suffix}',
    version=version,
    author='Bjorn H. Wehlin',
    license='Apache2',
    packages=['masspcf'],
    python_requires=">=3.10",
    cmake_args=['-DCMAKE_BUILD_TYPE=Release']
)
