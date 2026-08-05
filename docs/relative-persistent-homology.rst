============================
Relative Persistent Homology
============================

Relative persistent homology studies a filtered space *relative to* another:
instead of tracking the homology of a single growing complex, it tracks a map
between two filtrations through the kernels, images, and cokernels of the
induced maps on homology :footcite:`EdelsbrunnerRelativePersistence`. The
resulting barcodes record where the two filtrations *disagree*, rather than
what either one looks like on its own. The
:doc:`homological kernel <homological-kernel>` is the special case where the
two filtrations are built from two comparable distances :math:`d' \le d` on
the same point set, and only the kernel of the induced map is kept.


Background
==========

.. toctree::
   :maxdepth: 2

   homological-kernel


.. footbibliography::
