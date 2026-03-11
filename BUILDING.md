## Bump version number

- In pyproject.toml, change [project].version

## Minimal module build

Minimal module builds are now enabled whenever `SKBUILD` is not in use (i.e., plain CMake builds). In your IDE, just run a normal CMake configure/build/install and you can use `make` instead of a full `pip install`.

**Before doing this, you should run a manual `pip install .` first!**

## Building the documentation

Go to the `docs` directory and run `Make` on Linux/OSX, or `make.bat` on Windows.

This will create `_build` containing the documentation.
