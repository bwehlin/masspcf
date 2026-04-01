=================
Random Generation
=================

masspcf provides deterministic random generation that is reproducible
regardless of thread count or execution order. All random generation functions
accept an optional :py:class:`~masspcf.random.Generator` for explicit seed
control.


Generators
==========

A :py:class:`~masspcf.random.Generator` holds a seed used to derive
independent, deterministic random streams for each tensor element::

   import masspcf as mpcf

   gen = mpcf.random.Generator(seed=42)

Pass a generator to any function that produces random output::

   X = mpcf.random.noisy_sin((10, 20), generator=gen)

Two generators with the same seed always produce the same result::

   gen_a = mpcf.random.Generator(seed=123)
   gen_b = mpcf.random.Generator(seed=123)

   X = mpcf.random.noisy_sin((5, 10), generator=gen_a)
   Y = mpcf.random.noisy_sin((5, 10), generator=gen_b)
   # X and Y are identical

A generator can be re-seeded with :py:meth:`~masspcf.random.Generator.seed`::

   gen.seed(99)

Creating a generator without a seed uses a non-deterministic seed from the
operating system::

   gen = mpcf.random.Generator()  # non-deterministic

Global seed
-----------

For convenience, :py:func:`~masspcf.random.seed` seeds a global generator
used when no explicit generator is passed::

   mpcf.random.seed(42)
   A = mpcf.random.noisy_sin((10, 20))

   mpcf.random.seed(42)
   B = mpcf.random.noisy_sin((10, 20))
   # A and B are identical


Noisy trigonometric PCFs
========================

:py:func:`~masspcf.random.noisy_sin` and :py:func:`~masspcf.random.noisy_cos`
create tensors of piecewise constant functions that approximate
:math:`\sin(2\pi t)` and :math:`\cos(2\pi t)` with additive Gaussian noise::

   sines = mpcf.random.noisy_sin((200,), n_points=100)
   cosines = mpcf.random.noisy_cos((10, 50), n_points=30)

Each breakpoint :math:`t_i` is drawn uniformly from :math:`[0, 1]` and sorted,
with the first breakpoint fixed at :math:`t = 0`. The value at each breakpoint
is :math:`f(t_i) + \varepsilon_i` where :math:`\varepsilon_i \sim \mathcal{N}(0, 0.1)`.
The last value is always set to zero.

Pass ``dtype=mpcf.pcf64`` for 64-bit precision (the default is ``pcf32``).


How determinism works
=====================

Each element in the output tensor is assigned a deterministic sub-seed derived
from the master seed and the element's flat (row-major) index. This means:

- The same seed always produces the same tensor, even across different runs.
- The result is independent of how many threads are used or the order in which
  elements are computed.
- Different elements receive independent random streams.
