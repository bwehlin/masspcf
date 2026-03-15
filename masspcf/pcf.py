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

from . import _mpcf_cpp as cpp
from .typing import float32, float64, pcf32, pcf64, f32, f64, _validate_dtype, _check_deprecated_dtype, _assert_valid_dtype

class Pcf:
  r"""A piecewise constant function (PCF).

  A PCF is defined by a sequence of (time, value) pairs
  :math:`(t_0, v_0), (t_1, v_1), \ldots, (t_{n-1}, v_{n-1})` with
  :math:`t_0 = 0` and :math:`t_0 < t_1 < \cdots < t_{n-1}`. The function
  takes the value :math:`v_i` on the interval :math:`[t_i, t_{i+1})` for
  :math:`0 \leq i < n-1`, and :math:`v_{n-1}` on :math:`[t_{n-1}, \infty)`.

  Parameters
  ----------
  arr : numpy.ndarray or Pcf or list
      Input data. If an ndarray or list, should have shape (n, 2) where each
      row is a (time, value) pair. Can also be an existing ``Pcf`` to copy.
  dtype : type, optional
      Data type for the PCF (``pcf32`` or ``pcf64``). If ``None``, the dtype
      is inferred from the input array (e.g. a ``numpy.float32`` array
      produces a 32-bit PCF).

  Examples
  --------
  >>> import numpy as np
  >>> import masspcf as mpcf
  >>> f = mpcf.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]], dtype=np.float32))
  >>> f.size()
  3
  """
  def __init__(self, arr : np.ndarray, dtype=None):
    if isinstance(arr, Pcf):
      self._data = arr._data
      self.ttype = arr.ttype
      self.vtype = arr.vtype

    elif isinstance(arr, np.ndarray):
      if dtype is not None:
        dtype = _check_deprecated_dtype(dtype)

        if dtype == pcf32:
          dtype = np.float32
        elif dtype == pcf64:
          dtype = np.float64

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
    """Convert the PCF to a numpy array of shape (n, 2) with (time, value) rows."""
    return np.asarray(self._data)

  def _debug_print(self):
    self._data.debugPrint()

  def astype(self, dtype):
    """Return a copy of the PCF cast to the given dtype (``pcf32`` or ``pcf64``)."""
    _assert_valid_dtype(dtype, [pcf32, pcf64, np.float32, np.float64])
    np_dtype = {pcf32: np.float32, pcf64: np.float64}.get(dtype, dtype)
    return Pcf(self.to_numpy().astype(np_dtype))

  def __add__(self, rhs):
    if not _has_matching_types(self, rhs):
      raise TypeError('Mismatched PCF types')

    temp = self._data.copy()
    params = (temp, rhs._data)

    if _has_matching_types(self, tPcf_f32_f32):
      return Pcf(cpp.Backend_f32_f32.add(*params))
    elif _has_matching_types(self, tPcf_f64_f64):
      return Pcf(cpp.Backend_f64_f64.add(*params))

    raise TypeError(f'Unsupported PCF type for addition ({type(self._data).__name__})')
  
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

  def __array__(self, dtype=None, copy=None):
    arr = np.asarray(self._data)
    if dtype is not None:
      arr = arr.astype(dtype, copy=False)
    return arr

  def __eq__(self, other):
    return np.array_equal(self.__array__(), np.asarray(other))


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


