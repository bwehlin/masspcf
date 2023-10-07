from shlex import join
from . import mpcf_cpp as cpp
import numpy as np
import asyncio

class Rectangle:
  def __init__(self, left, right, top, bottom, dtype=np.float32):
    if dtype == np.float32:
      self.data_ = cpp.Rectangle_f32_f32(np.float32(left), np.float32(right), np.float32(top), np.float32(bottom))
    elif dtype == np.float64:
      self.data_ = cpp.Rectangle_f64_f64(np.float64(left), np.float64(right), np.float64(top), np.float64(bottom))
    else:
      raise TypeError("dtype must be either np.float32 or np.float64")
  
  @property
  def left(self):
    return self.data_.left
  
  @left.setter
  def left(self, v):
    self.data_.left = v
  
  @property
  def right(self):
    return self.data_.right
  
  @right.setter
  def right(self, v):
    self.data_.right = v

  @property
  def top(self):
    return self.data_.top
  
  @top.setter
  def top(self, v):
    self.data_.top = v

  @property
  def bottom(self):
    return self.data_.bottom
  
  @bottom.setter
  def bottom(self, v):
    self.data_.bottom = v



class Pcf:
  def __init__(self, arr):
    if isinstance(arr, np.ndarray):
      if arr.dtype == np.float32:
        self.data_ = cpp.Pcf_f32_f32(arr)
      elif arr.dtype == np.float64:
        self.data_ = cpp.Pcf_f64_f64(arr)
    elif isinstance(arr, cpp.Pcf_f32_f32):
      self.data_ = arr
    elif isinstance(arr, cpp.Pcf_f64_f64):
      self.data_ = arr

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

    params = (self.data_, rhs.data_)

    if _has_matching_types(self, tPcf_f32_f32):
      self.data_ = cpp.Backend_f32_f32.add(*params)
    elif _has_matching_types(self, tPcf_f64_f64):
      self.data_ = cpp.Backend_f64_f64.add(*params)
    
    return self
  
  def __truediv__(self, c):
    self.data_ = self.data_.div_scalar(c)
    return self

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
  sz = backend.get_input_size(fsdata)
  #if len(fsdata) < 100 or sz < 10000:
  #  return Pcf(backend.mem_average(fsdata, 8))
  #print(f'In size: {sz}')
  future = backend.async_average(executor, fsdata)
  while future.wait_for(50) != cpp.FutureStatus.ready:
    pass #print('HELLO')
  #print('Wait finished')
  #print(fsdata)
  return Pcf(future.get())

def async_matrix_l1_dist(fs):
  fsdata, backend = _prepare_list(fs)
  dtype = np.float64 #fsdata[0].to_numpy().dtype
  n = len(fs)
  matrix = np.zeros((n, n), dtype=dtype, order='c')
  future = backend.async_matrix_l1_dist(executor, matrix, fsdata)
  while future.wait_for(50) != cpp.FutureStatus.ready:
    pass #print('HELLO')
  return matrix


async def async_mem_average(fs, chunksz=8):
  fsdata, backend = _prepare_list(fs)
  print('Begin await')
  await backend.async_mem_average(fsdata, chunksz)
  print('After await')
  #return fut
  
executor = cpp.Executor()

def back_average(fs):
  fsdata, backend = _prepare_list(fs)
  import time
  future = backend.spawn_pcf(executor, lambda : backend.mem_average(fsdata, 200))
  #future = backend.spawn_avg(executor, fsdata)
  #print(future.wait_for(1))
  while True:
    future.wait_for(100)
    status = future.get_last_status()
    print(status)
    time.sleep(0.5)
    if status == cpp.FutureStatus.ready:
      break
  #while future.wait_for(10) != cpp.FutureStatus.ready:
  #  print('HELLO')
  #print(future.wait_for(1))
  print('DONE')

  return
  print(fut)
  #job = backend.spawn_average(executor, fsdata)
  job = cpp.BackgroundJob(executor, lambda : backend.mem_average(fsdata, 2))
  while job.done() == False:
    job.wait_for(100)
    print('Hello Darkness')
  print('Done')

def mem_average(fs, chunksz=8):
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.mem_average(fsdata, chunksz))

def st_average(fs):
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.st_average(fsdata))

def parallel_reduce(fs, cb):
  cb = cb.address
  fsdata, backend = _prepare_list(fs)
  return Pcf(backend.parallel_reduce(fsdata, cb))

def l1_inner_prod(fs):
  fsdata, backend = _prepare_list(fs)
  return backend.l1_inner_prod(fsdata)

def matrix_l1_dist(fs):
  fsdata, backend = _prepare_list(fs)
  return backend.matrix_l1_dist(fsdata)
