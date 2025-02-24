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
from .pcf import _prepare_list, Pcf
from .array import Array, View, Container
from .typing import float32, float64

import numpy as np
from tqdm import tqdm

def _wait_for_task(task, verbose=True):
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
    _wait_for_task(task, verbose=verbose)
  finally:
    if task is not None:
      task.request_stop()
      _wait_for_task(task, verbose=verbose)
  
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
    _wait_for_task(task, verbose=verbose)
  finally:
    if task is not None:
      task.request_stop()
      _wait_for_task(task, verbose=verbose)

  return matrix

def pdist(fs : Container, p=1, verbose=False):
  r"""Compute pairwise (:math:`L_p`) distances between all PCFs in a 1-dimensional array. That is, if `fs` is the array :math:`\begin{pmatrix} f_1 & f_2 & \cdots & f_n \end{pmatrix}`, we compute the matrix

  .. math::
    \begin{pmatrix}
      d_{11} & d_{12} & \cdots & d_{1n} \\
      d_{21} & d_{22} & \cdots & d_{2n} \\
      \vdots & \vdots & \ddots & \vdots & \\
      d_{n1} & d_{n2} & \cdots & d_{nn} \\
    \end{pmatrix},
  
  where
  
  .. math::
    d_{ij} = \left(\int_0^\infty |f_i(t)-f_j(t)|^p\, dt\right)^{1/p}.

  Parameters
  ----------
  fs : Container
      1-d array of PCFs whose pairwise distances are to be computed.
  p : int, optional
      :math:`p` parameter in the :math:`L_p` distance, by default 1
  verbose : bool, optional
      Print additional information during the computation, by default False
  
  Returns
  -------
  numpy.ndarray
    If input is a container of size `n`, the returned `numpy.ndarray` has shape `(n,n)`. The `i,j`-th entry is the distance from PCF `i` to PCF `j` in the input container.
  """
  def task_factory(backend, matrix, buf):
    if p == 1:
      return backend.calc_pdist_1(matrix, buf)
    else:
      return backend.calc_pdist_p(matrix, buf, p)

  return _compute_matrix(fs, task_factory, verbose)


def l2_kernel(fs : Container, verbose=False):
  r"""Compute the :math:`L_2` kernel (Gram) matrix of all PCFs in a 1-dimensional array. That is, if `fs` is the array :math:`\begin{pmatrix} f_1 & f_2 & \cdots & f_n \end{pmatrix}`, we compute the matrix

  .. math::
    \begin{pmatrix}
      g_{11} & g_{12} & \cdots & g_{1n} \\
      g_{21} & g_{22} & \cdots & g_{2n} \\
      \vdots & \vdots & \ddots & \vdots & \\
      g_{n1} & g_{n2} & \cdots & g_{nn} \\
    \end{pmatrix},
  
  where
  
  .. math::
    g_{ij} = \int_0^\infty f_i(t)f_j(t) \, dt.

  Parameters
  ----------
  fs : Container
      1-d array of PCFs on which the kernel is to be computed.
  verbose : bool, optional
      Print additional information during the computation, by default False
  
  Returns
  -------
  numpy.ndarray
    If input is a container of size `n`, the returned `numpy.ndarray` has shape `(n,n)`. The `i,j`-th entry is the distance from PCF `i` to PCF `j` in the input container.
  """
  def task_factory(backend, matrix, buf):
    return backend.calc_l2_kernel(matrix, buf)

  return _compute_matrix(fs, task_factory, verbose)
