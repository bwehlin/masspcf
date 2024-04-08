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
from .pcf import _prepare_list, Pcf
from .array import Array, View
from .typing import float32, float64

import numpy as np
from tqdm import tqdm

def wait_for_task(task, verbose=True):
  def init_progress(task):
    progress = tqdm(total=task.work_total(), unit_scale=True, unit=task.work_step_unit(), desc=task.work_step_desc())
    return progress

  if verbose:
    progress = init_progress(task)
    work_step = task.work_step()

  wait_time_ms = 50
  while not task.wait_for(wait_time_ms):
    if verbose:
      progress.update(task.work_completed() - progress.n)
      new_work_step = task.work_step()
      if new_work_step != work_step:
        work_step = new_work_step
        print('')
        progress = init_progress(task)
    
  if verbose:
    progress.update(task.work_completed() - progress.n)


def _compute_matrix2(fs : list[Pcf], task_factory, verbose):
  if len(fs) == 0:
      return np.zeros((0,0))

  fsdata, backend = _prepare_list(fs)
  dtype = fs[0].vtype
  
  n = len(fs)

  #shape = ((n*(n-1)) / 2,) if condensed else (n, n)
  shape = (n,n)
  matrix = np.zeros(shape, dtype=dtype, order='c')

  task = None
  try:
    task = task_factory(backend, matrix, fsdata)
    wait_for_task(task, verbose=verbose)
  finally:
    if task is not None:
      task.request_stop()
      wait_for_task(task, verbose=verbose)
  
  return matrix

def _compute_matrix(fs, task_factory, verbose=False):
  if fs.dtype == float32:
    backend = cpp.Backend_f32_f32
    npdtype = np.float32
  else:
    backend = cpp.Backend_f64_f64
    npdtype = np.float64

  matrix = np.zeros((fs.shape[0], fs.shape[0]), dtype=npdtype)
  buf = fs._as_view().data.strided_buffer()

  task = None
  try:
    task = task_factory(backend, matrix, buf)
    wait_for_task(task, verbose=verbose)
  finally:
    if task is not None:
      task.request_stop()
      wait_for_task(task, verbose=verbose)

  return matrix

def pdist(fs, p=1, verbose=False):
  def task_factory(backend, matrix, buf):
    return backend.calc_pdist(matrix, buf)

  return _compute_matrix(fs, task_factory, verbose)


def l2_kernel(fs : list[Pcf], verbose=True):
  def task_factory(backend, matrix, fsdata):
    return backend.matrix_l2_kernel(matrix, fsdata)
  
  return _compute_matrix(fs, task_factory, verbose)
