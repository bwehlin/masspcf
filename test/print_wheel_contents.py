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

import zipfile, os, glob, subprocess

wheel_dir=os.environ['CIBW_REPAIRED_WHEEL_DIR']
project_name=os.environ['CIBW_PROJECT_NAME']

whl_files=glob.glob(os.path.join(wheel_dir, f'{project_name}-*.whl'))
whl_path=whl_files[0]

print(f'Contents of {whl_path}:')

zf=zipfile.ZipFile(whl_path,'r')

[print(f) for f in zf.namelist()]

zf.close()
