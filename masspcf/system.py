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


def force_cpu(on : bool):
  """ Set forced execution on CPU. By default, execution may happen on either CPU or GPU (if using a GPU-enabled build of masspcf).

  Parameters
  ----------
  on : bool
    If `True`, force execution on CPU for all operations. If `False`, execution may happen on either CPU or GPU (if using a GPU-enabled build of masspcf).
  """
  cpp.force_cpu(on)

def set_block_size(x : int, y : int):
  """ Set CUDA block size for (GPU) matrix computations. This is an advanced option that should only be modified by expert users.
  
  Parameters
  ----------
  x : int
    Horizontal block size
  
  y : int
    Vertical block size
  """
  cpp.set_block_dim(x, y)

def limit_cpus(n : int):
  """ Sets the upper limit on the number of CPU threads that can be used for computations. 
  
  Typically, the default corresponding to the number of hardware CPU threads is a good choice but it can be warranted to limit the number of threads in, e.g., multi-user environments. For normal use, we recommend using the default.

  Parameters
  ----------
  n : int
    Number of CPU threads to use
  """
  cpp.limit_cpus(n)

def limit_gpus(n : int):
  """ Sets the number of GPUs that can be used by masspcf. By default, all available GPUs are used.
  
  This option only has an effect if masspcf is compiled with GPU support.

  Parameters
  ----------
  n : int
    Number of GPUs to use
  """
  cpp.limit_gpus(n)

def set_cuda_threshold(n : int):
  """ Sets how many PCFs are required in a matrix computation before computations are moved from CPU to GPU. By default, the threshold is set to 500 PCFs.
  
  Parameters
  ----------
  n : int
    Number of PCFs required before (supported) matrix computations are moved to GPU
  """
  cpp.set_cuda_threshold(n)

def set_device_verbose(on : bool):
  """ Enable verbose device output. In this mode, when operations that may occur on GPU are invoked, a message is logged stating whether the operation will be performed on CPU or GPU.

  Parameters
  ----------
  on : bool
    Enable verbose device logging
  """
  cpp.set_device_verbose(on)

