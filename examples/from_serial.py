import masspcf as mpcf
from masspcf.plotting import plot as plotpcf
import numpy as np
import matplotlib.pyplot as plt

content = [[0, 2.5], [1.5, 1.2], [3.14, 0],
           [0, 7.0], [3.8, 5.5], [4.5, 1.5], [7, 0],
           [0, 3], [2, 0]]

enumeration = [[0, 3], [3, 7], [7, 9]]

content = np.array(content)
enumeration = np.array(enumeration)

print(content.shape)
print(enumeration.shape)

F = mpcf.from_serial_content(content, enumeration)
plotpcf(F)
plt.show()
