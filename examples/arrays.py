import masspcf as mpcf

z = mpcf.zeros((3, 4))
print(z.shape())

print(z[:,:2])
v = z[0,:]
print(v.shape())