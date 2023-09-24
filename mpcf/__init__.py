import sys
import os

directory = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(directory + '/../build')

import mpcf_cpp