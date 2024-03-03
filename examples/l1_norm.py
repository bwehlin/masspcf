from masspcf import Pcf, l1_norm

from masspcf.plotting import plot as plotpcf
import matplotlib.pyplot as plt
import numpy as np


# Construct three PCFs of increasing complexity
f0 = Pcf(np.array([[0.], [0.]]))
f1 = Pcf(np.array([[0., 1.5], [2., 0.]]))
f2 = Pcf(np.array([[0., 1.5, 3., 4.], [2., 1., 7., 0.]]))

# f0 will not show up on the plot since it's a single point
plotpcf(f0, label='f0')
plotpcf(f1, label='f1')
plotpcf(f2, label='f2')
plt.legend()

# masspcf supports l1 norm for lists of PCFs and single PCFs
print(l1_norm([f0, f1, f2]))
print(l1_norm(f1))
