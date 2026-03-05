import sys
import zipfile
import io
from pathlib import Path
from fnmatch import fnmatch

IGNORE_GLOBS = [
    '*/masspcf/_mpcf_cpp*.so*',             # Linux/Mac Python extension module
    '*/masspcf/_mpcf_cpp*.pyd',             # Windows Python extension module
    '*/sboms/auditwheel.cdx.json',          # auditwheel metadata, varies by platform
    'masspcf_cpu.libs/libgcc*.so*',         # bundled gcc library
    'masspcf_cpu.libs/libstdc++*.so*',      # bundled stdc++ library
]

def should_ignore(filename):
    return any(fnmatch(filename, pattern) for pattern in IGNORE_GLOBS)

def read_wheel(whl):
    all_files = whl.namelist()
    raw = sorted(f for f in all_files if not f.endswith('/'))
    filtered = {
        f for f in all_files
        if not should_ignore(f) and not f.endswith('/')
    }
    return filtered, raw

def get_wheels_from_zip(zip_path):
    wheels = {}
    raw = {}
    with zipfile.ZipFile(zip_path) as zf:
        wheel_names = [n for n in zf.namelist() if n.endswith('.whl')]
        for wheel_name in wheel_names:
            with zf.open(wheel_name) as wf:
                wheel_data = io.BytesIO(wf.read())
                with zipfile.ZipFile(wheel_data) as whl:
                    wheels[Path(wheel_name).name], raw[Path(wheel_name).name] = read_wheel(whl)
    return wheels, raw

def get_wheels_from_dir(dir_path):
    wheels = {}
    raw = {}
    for wheel_path in sorted(Path(dir_path).glob('*.whl')):
        with zipfile.ZipFile(wheel_path) as whl:
            wheels[wheel_path.name], raw[wheel_path.name] = read_wheel(whl)
    return wheels, raw

def dump_contents(name, files):
    print(f"  Contents of {name}:")
    for f in files:
        print(f"    {f}")

def main(input_path, verbose=False):
    path = Path(input_path)
    if path.is_dir():
        wheels, raw = get_wheels_from_dir(path)
    elif path.suffix == '.zip':
        wheels, raw = get_wheels_from_zip(path)
    else:
        print(f"Error: {input_path} is not a directory or a .zip file")
        return 1

    names = list(wheels.keys())

    if len(names) < 2:
        print("Need at least 2 wheels to compare")
        return 1

    reference_name = names[0]
    reference = wheels[reference_name]

    print(f"Reference: {reference_name}\n")

    if verbose:
        dump_contents(reference_name, raw[reference_name])
        print()

    failed = False
    for name in names[1:]:
        files = wheels[name]
        only_in_ref = reference - files
        only_in_this = files - reference

        if not only_in_ref and not only_in_this:
            print(f"✅ {name}: identical file list")
        else:
            failed = True
            print(f"❌ {name}:")
            for f in sorted(only_in_ref):
                print(f"   only in reference: {f}")
            for f in sorted(only_in_this):
                print(f"   only in this:      {f}")
            if verbose:
                dump_contents(name, raw[name])
        print()

    return 1 if failed else 0

if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = '--verbose' in args or '-v' in args
    paths = [a for a in args if not a.startswith('-')]

    if len(paths) != 1:
        print("Usage: python compare_wheels.py [-v|--verbose] <wheels.zip|wheels_dir/>")
        sys.exit(1)

    sys.exit(main(paths[0], verbose=verbose))