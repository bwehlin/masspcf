import masspcf as mpcf

z = mpcf.zeros((3, 4, 6))
print(z.shape())

#print(z[:,:2])
v = z[0, 1:3, :]


print(v.shape())
print(v.data)
v1 = v[0,1:3]
print(v1.shape())
print(v1.data)
v2 = v1[1:2]
print(v2.data)

#print(v)