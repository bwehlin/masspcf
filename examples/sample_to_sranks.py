from masspcf.sample_to_sranks import sample_to_sranks
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

N = 50
dim = 3

data = np.random.randn(N, dim)
sample = np.random.randint(0, N, (N, 2, 25))

X = data[sample[0, 0, :], :]

dgms = ripser(X)['dgms']
print(dgms)
#plot_diagrams(dgms, lifetime=True)
#plt.show()

#print(data)
print(sample.shape)

fs = sample_to_sranks(data, sample)
print(fs)

from masspcf.plotting import plot as pcfplot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

pcfplot(fs[:,0,0], ax=ax[0])
plot_diagrams(dgms, ax=ax[1], lifetime=True)
plt.show()