## Bump version number

- In CMakeLists.txt, change ${PROJ_VERSION} (e.g., set(PROJ_VERSION "0.4.0"))
- In pyproject.toml, change [project].version

## Minimal module build

In your IDE, it's recommended to have the environment variable MINIMAL_MODULE_BUILD=1 set. With this enabled, building and installing masspcf can be done via "make" rather than a full pip install. 

**Before doing this, you should run a manual pip install . first!**  