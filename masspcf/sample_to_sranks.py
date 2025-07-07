#    Copyright 2024-2025 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from . import mpcf_cpp as cpp
from .array import Array

def sample_to_sranks(data, sample, maxDim=1):
    return Array(cpp.sample_to_sranks_64(data, sample, maxDim))
