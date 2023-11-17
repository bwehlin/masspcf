from skbuild import setup

setup(
    name='masspcf',
    version='0.1.0',
    description='Massively parallel computations for piecewise constant functions',
    author='Bjorn H. Wehlin',
    license='TBD',
    packages=['masspcf'],
    python_requires=">=3.9",
    cmake_args=['-DCMAKE_BUILD_TYPE=Release'],
    install_requires=[
        'numpy', 'numba', 'tqdm'
    ]
)
