#    Copyright 2024 Bjorn Wehlin
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

def force_cpu(on : bool):
  cpp.force_cpu(on)

def set_block_size(x : int, y : int):
  cpp.set_block_dim(x, y)

def limit_cpus(n : int):
  cpp.limit_cpus(n)

def limit_gpus(n : int):
  cpp.limit_gpus(n)

def set_cuda_threshold(n : int):
  cpp.set_cuda_threshold(n)

def set_device_verbose(on : bool):
  cpp.set_device_verbose(on)

