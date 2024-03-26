import masspcf as mpcf

z = mpcf.zeros((3, 4,5))
print(z.shape())

print(z[:,:2])
v = z[0,1:3,:-2]
print(v.shape())