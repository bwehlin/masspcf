=================
Random Generation
=================

The :py:mod:`masspcf.random` module generates tensors of noisy trigonometric
PCFs, useful for quick experimentation and testing. Results are fully
deterministic when seeded, regardless of thread count or execution order.


Generating noisy PCFs
=====================

:py:func:`~masspcf.random.noisy_sin` and :py:func:`~masspcf.random.noisy_cos`
create tensors of piecewise constant functions that approximate
:math:`\sin(2\pi t)` and :math:`\cos(2\pi t)` with additive Gaussian noise::

   import masspcf as mpcf

   # 200 noisy sin(2*pi*t) functions, each with 100 breakpoints
   sines = mpcf.random.noisy_sin((200,), n_points=100)

   # 2-D: 10 x 50 noisy cosine functions
   cosines = mpcf.random.noisy_cos((10, 50), n_points=30)

Each breakpoint :math:`t_i` is drawn uniformly from :math:`[0, 1]` and sorted,
with the first breakpoint fixed at :math:`t = 0`. The value at each breakpoint
is :math:`f(t_i) + \varepsilon_i` where :math:`\varepsilon_i \sim \mathcal{N}(0, 0.1)`.
The last value is always set to zero.

Pass ``dtype=mpcf.pcf64`` for 64-bit precision (the default is ``pcf32``).


Seeding for reproducibility
===========================

Global seed
-----------

:py:func:`~masspcf.random.seed` seeds the global random number generator.
Subsequent calls with the same seed produce identical output::

   mpcf.random.seed(42)
   A = mpcf.random.noisy_sin((10, 20))

   mpcf.random.seed(42)
   B = mpcf.random.noisy_sin((10, 20))
   # A and B are identical

User-provided generators
------------------------

For finer control, create a :py:class:`~masspcf.random.Generator` and pass it
to any generation function. This avoids mutating global state and makes it easy
to produce independent reproducible streams::

   gen1 = mpcf.random.Generator(seed=42)
   gen2 = mpcf.random.Generator(seed=99)

   A = mpcf.random.noisy_sin((10, 20), generator=gen1)
   B = mpcf.random.noisy_cos((10, 20), generator=gen2)

Two generators created with the same seed always produce the same sequence::

   gen_a = mpcf.random.Generator(seed=123)
   gen_b = mpcf.random.Generator(seed=123)

   X = mpcf.random.noisy_sin((5, 10), generator=gen_a)
   Y = mpcf.random.noisy_sin((5, 10), generator=gen_b)
   # X and Y are identical

A generator can be re-seeded::

   gen = mpcf.random.Generator(seed=42)
   gen.seed(99)  # change the seed

Creating a ``Generator`` without a seed uses a non-deterministic seed from the
operating system::

   gen = mpcf.random.Generator()  # non-deterministic


How determinism works
=====================

Each element in the output tensor is assigned a deterministic sub-seed derived
from the master seed and the element's flat (row-major) index. This means:

- The same seed always produces the same tensor, even across different runs.
- The result is independent of how many threads are used or the order in which
  elements are computed.
- Different elements receive independent random streams.
