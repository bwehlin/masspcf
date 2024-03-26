import masspcf as mpcf

z = mpcf.zeros((3, 4,5))
print(z.shape())

#print(z[:,:2])
v = z[0,1:3,:-2]
print(v.shape())
v1 = v[0,:]
print(v1.shape())

#print(v)