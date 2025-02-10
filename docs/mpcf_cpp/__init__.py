# This is to avoid having to recompile on every documentation update. We provide a fake mpcf_cpp so that there are no fatal errors when 'masspcf' imports mpcf_cpp. We also have to provide implementation stubs for some of the classes.

class Pcf_f32_f32:
    def __init__(self, arr):
        pass

class Pcf_f64_f64:
    def __init__(self, arr):
        pass

class Backend_f32_f32:
    pass

class Backend_f64_f64:
    pass

