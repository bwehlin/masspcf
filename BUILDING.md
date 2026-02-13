## Bump version number

- In pyproject.toml, change [project].version

## Minimal module build

In your IDE, it's recommended to have the environment variable MINIMAL_MODULE_BUILD=1 set. With this enabled, building and installing masspcf can be done via "make" rather than a full pip install. 

**Before doing this, you should run a manual pip install . first!**  

## Building the documentation

Go to the `docs` directory and run `Make` on Linux/OSX, or `make.bat` on Windows.

This will create `_build` containing the documentation.