#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

print('!!!!!!!!!! Printing wheel contents !!!!!!!!!!')

import os
import masspcf
import importlib.util

# Fallback for compiled modules with no __file__
pkg_path = getattr(masspcf, "__file__", None)
if pkg_path is None:
    spec = importlib.util.find_spec("masspcf")
    pkg_path = spec.origin if spec and spec.origin else None

if pkg_path is None:
    print("Cannot determine installed package path for masspcf")
else:
    pkg_dir = os.path.dirname(pkg_path)
    print(f"Contents of installed package masspcf at {pkg_dir}:")
    for dp, dn, filenames in os.walk(pkg_dir):
        for f in filenames:
            print(os.path.join(dp, f))
