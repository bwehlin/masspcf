#    Copyright 2024-2026 Bjorn Wehlin
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

from typing import Union
import numpy as np

from . import mpcf_cpp as cpp
from .typing import float32, float64

__all__ = ['Pcf']

class Pcf:
  """
  Piecewise Constant Function

  Parameters
  ----------
  arr : (n x 2) array of values, first
  """
  def __init__(self, arr : np.ndarray, dtype=None):
    if isinstance(arr, Pcf):
      self._data = arr._data
      self.ttype = arr.ttype
      self.vtype = arr.vtype

    elif isinstance(arr, np.ndarray):
      if dtype is not None:
          if arr.dtype != dtype:
              arr = arr.astype(dtype)

      if arr.dtype == np.float32:
        self._data = cpp.Pcf_f32_f32(arr)
      elif arr.dtype == np.float64:
        self._data = cpp.Pcf_f64_f64(arr)
      elif arr.dtype == np.int64:
        arr = arr.astype(np.float64)
        self._data = cpp.Pcf_f64_f64(arr)
      elif arr.dtype == np.int32:
        arr = arr.astype(np.float32)
        self._data = cpp.Pcf_f32_f32(arr)
      else:
        raise ValueError('Unsupported array type (must be np.float32/64 or np.int32/64)')

      self.ttype = arr.dtype
      self.vtype = arr.dtype
    elif isinstance(arr, cpp.Pcf_f32_f32):
      self._data = arr
      self.ttype = np.float32
      self.vtype = np.float32
    elif isinstance(arr, cpp.Pcf_f64_f64):
      self._data = arr
      self.ttype = np.float64
      self.vtype = np.float64
    elif isinstance(arr, list):
      if dtype is None:
        dtype = np.float32
      data = np.array(arr, dtype=dtype)
      if dtype == np.float32:
        self._data = cpp.Pcf_f32_f32(data)
        self.ttype = np.float32
        self.vtype = np.float32
      elif dtype == np.float64:
        self._data = cpp.Pcf_f64_f64(data)
        self.ttype = np.float64
        self.vtype = np.float64
      else:
        raise ValueError('Unsupported dtype')
        
    else:
      raise ValueError(f'Tried to create PCF from unsupported input data of type {type(arr)}.')

  def _get_time_type(self):
    return self._data.get_time_type()

  def _get_value_type(self):
    return self._data.get_value_type()

  def _get_time_value_type(self):
    return self._get_time_type() + '_' + self._get_value_type()

  def to_numpy(self):
    return np.array(self._data)

  def _debug_print(self):
    self._data.debugPrint()

  def astype(self, dtype):
    return Pcf(self.to_numpy().astype(dtype))

  def __add__(self, rhs):
    if not _has_matching_types(self, rhs):
      raise TypeError('Mismatched PCF types')

    temp = self._data.copy()
    params = (temp, rhs._data)

    if _has_matching_types(self, tPcf_f32_f32):
      return Pcf(cpp.Backend_f32_f32.add(*params))
    elif _has_matching_types(self, tPcf_f64_f64):
      return Pcf(cpp.Backend_f64_f64.add(*params))
    
    return self
  
  def __truediv__(self, c):
    self._data = self._data.div_scalar(c)
    return self
  
  def size(self):
    return self._data.size()
  
  #def save(self):
  #  return self._data.to_numpy().save()

  def __str__(self):
    dtname = 'float32' if self.vtype is np.float32 else 'float64' # TODO: mixed vtype/ttype?
    return f'<PCF size={self._data.size()}, dtype={dtname}>'

  def __array__(self):
    return np.array(self._data)

  def __eq__(self, other):
    return np.array_equal(np.array(self), np.array(other))


tPcf_f32_f32 = Pcf(np.array([[0, 0]]).astype(np.float32))
tPcf_f64_f64 = Pcf(np.array([[0, 0]]).astype(np.float64))

backend_f32_f32 = cpp.Backend_f32_f32
backend_f64_f64 = cpp.Backend_f64_f64

def _is_pcf(val):
  return isinstance(val, Pcf)

def _is_pcf_data(val):
  return isinstance(val, cpp.Pcf_f32_f32) or isinstance(val, cpp.Pcf_f64_f64)

def _is_convertible_to_pcf_data(val):
  return _is_pcf(val) or _is_pcf_data(val)

def _pcf_as_data(val):
  if _is_pcf(val):
    return val._data
  else:
    return val

def _get_backend(f : Pcf):
  if _has_matching_types(f, tPcf_f32_f32):
    return backend_f32_f32
  elif _has_matching_types(f, tPcf_f64_f64):
    return backend_f64_f64
  else:
    raise TypeError("Unknown PCF type")

def _has_matching_types(f : Pcf, g : Pcf):
  return type(f._data) == type(g._data)

def _get_dtype_from_data(data):
  if isinstance(data, cpp.Pcf_f32_f32):
    return float32
  elif isinstance(data, cpp.Pcf_f64_f64):
    return float64
  else:
    raise TypeError('Called with invalid type for this function')

def _prepare_list(fs):
  fsdata = [None]*len(fs)
  for i, f in enumerate(fs):
    fsdata[i] = f._data
    _ensure_same_type(fs[0], fs[i])
  backend = _get_backend(fs[0])
  return fsdata, backend

def _ensure_same_type(f : Pcf, g : Pcf):
  if not _has_matching_types(f, g):
    raise TypeError('Mismatched PCF types')

def combine(f : Pcf, g : Pcf, cb):
  _ensure_same_type(f, g)
  backend = _get_backend(f)
  return Pcf(backend.combine(f._data, g._data, cb))
  
def average(fs):
  """ Compute the average of a list of PCFs """

  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.average(fsdata))

def parallel_reduce(fs, cb):
  cb = cb.address
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.parallel_reduce(fsdata, cb))


