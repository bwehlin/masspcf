from skbuild import setup

setup(
    name='mpcf',
    version='0.0.1',
    description='Massively parallel computations for piecewise constant functions',
    author='Bjorn H. Wehlin',
    license='TBD',
    packages=['mpcf'],
    python_requires=">=3.9",
    cmake_args=['-DCMAKE_BUILD_TYPE=Release'],
    install_requires=[
        'numpy', 'numba', 'tqdm'
    ]
)
