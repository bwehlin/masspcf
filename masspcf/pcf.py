'''
    Copyright 2024 Bjorn Wehlin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

from . import mpcf_cpp as cpp
import numpy as np

from typing import Union

__all__ = ['Pcf']


class Pcf:
  def __init__(self, arr, dtype=None):
    if isinstance(arr, np.ndarray):
      if dtype is not None:
          if arr.dtype != dtype:
              arr = arr.astype(dtype)

      if arr.dtype == np.float32:
        self.data_ = cpp.Pcf_f32_f32(arr)
      elif arr.dtype == np.float64:
        self.data_ = cpp.Pcf_f64_f64(arr)
      elif arr.dtype == np.int64:
        arr = arr.astype(np.float64)
        self.data_ = cpp.Pcf_f64_f64(arr)
      elif arr.dtype == np.int32:
        arr = arr.astype(np.float32)
        self.data_ = cpp.Pcf_f32_f32(arr)
      else:
        raise ValueError('Unsupported array type (must be np.float32/64 or np.int32/64)')

      self.ttype = arr.dtype
      self.vtype = arr.dtype
    elif isinstance(arr, cpp.Pcf_f32_f32):
      self.data_ = arr
      self.ttype = np.float32
      self.vtype = np.float32
    elif isinstance(arr, cpp.Pcf_f64_f64):
      self.data_ = arr
      self.ttype = np.float64
      self.vtype = np.float64
    else:
      raise ValueError('Unsupported type')

  def get_time_type(self):
    return self.data_.get_time_type()

  def get_value_type(self):
    return self.data_.get_value_type()

  def get_time_value_type(self):
    return self.get_time_type() + '_' + self.get_value_type()

  def to_numpy(self):
    return np.array(self.data_.to_numpy())

  def debug_print(self):
    self.data_.debugPrint()

  def __add__(self, rhs):
    if not _has_matching_types(self, rhs):
      raise TypeError('Mismatched PCF types')

    temp = self.data_.copy()
    params = (temp, rhs.data_)

    if _has_matching_types(self, tPcf_f32_f32):
      return Pcf(cpp.Backend_f32_f32.add(*params))
    elif _has_matching_types(self, tPcf_f64_f64):
      return Pcf(cpp.Backend_f64_f64.add(*params))
    
    return self
  
  def __truediv__(self, c):
    self.data_ = self.data_.div_scalar(c)
    return self
  
  def size(self):
    return self.data_.size()
  
  def save(self):
    return self.data_.to_numpy().save()
  
  def __array__(self):
    return np.array(self.data_)


tPcf_f32_f32 = Pcf(np.array([[0],[0]]).astype(np.float32))
tPcf_f64_f64 = Pcf(np.array([[0],[0]]).astype(np.float64))

backend_f32_f32 = cpp.Backend_f32_f32
backend_f64_f64 = cpp.Backend_f64_f64

def _get_backend(f : Pcf):
  if _has_matching_types(f, tPcf_f32_f32):
    return backend_f32_f32
  elif _has_matching_types(f, tPcf_f64_f64):
    return backend_f64_f64
  else:
    raise TypeError("Unknown PCF type")

def _has_matching_types(f : Pcf, g : Pcf):
  return type(f.data_) == type(g.data_)

def _prepare_list(fs):
  fsdata = [None]*len(fs)
  for i, f in enumerate(fs):
    fsdata[i] = f.data_
    _ensure_same_type(fs[0], fs[i])
  backend = _get_backend(fs[0])
  return fsdata, backend

def _ensure_same_type(f : Pcf, g : Pcf):
  if not _has_matching_types(f, g):
    raise TypeError('Mismatched PCF types')

def combine(f : Pcf, g : Pcf, cb):
  _ensure_same_type(f, g)
  backend = _get_backend(f)
  return Pcf(backend.combine(f.data_, g.data_, cb))
  
def average(fs):
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.average(fsdata))

def parallel_reduce(fs, cb):
  cb = cb.address
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.parallel_reduce(fsdata, cb))

def lp_norm(fs : Union[Pcf, list[Pcf]], p : Union[int, float]):
  if isinstance(fs, list):
  
    if len(fs) == 0:
        return np.zeros((0,0))

    fsdata, backend = _prepare_list(fs)

    out = np.zeros((len(fs),))
    backend.list_l1_norm(out, fsdata)

    return out
  
  elif isinstance(fs, Pcf):
    backend = _get_backend(fs)
    return backend.single_l1_norm(fs.data_)

def l1_norm(fs : list[Pcf]):
  return lp_norm(fs, 1)

def l2_norm(fs : list[Pcf]):
  return lp_norm(fs, 2)

def linfinity_norm(fs : list[Pcf]):
  return lp_norm(fs,  float('inf'))
