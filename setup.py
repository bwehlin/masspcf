from skbuild import setup

setup(
    name='masspcf',
    version='0.3.1',
    author='Bjorn H. Wehlin',
    license='TBD',
    packages=['masspcf'],
    python_requires=">=3.10",
    cmake_args=['-DCMAKE_BUILD_TYPE=Release']
)
