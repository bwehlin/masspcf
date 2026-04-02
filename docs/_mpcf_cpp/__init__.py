# Stub _mpcf_cpp for documentation builds.
# Replaces the real backend selector so that autodoc can import masspcf
# without the compiled C++ extension.


class _Mock:
    def __init__(self, *a, **kw):
        pass

    def __call__(self, *a, **kw):
        return _Mock()

    def __getattr__(self, name):
        return _Mock()

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False

    def __repr__(self):
        return "_Mock()"


def __getattr__(name):
    return _Mock()
