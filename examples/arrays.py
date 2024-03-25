import masspcf as mpcf

z = mpcf.zeros((3, 4))
print(z.shape())

print(z[0, ..., 1:2:4, 1])