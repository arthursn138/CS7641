import numpy as np

x = np.array(((1,2), (1,1), (0,1), (1,0), (3,2), (1,5), (5,0), (2,4), (3,4)))
y = np.array(((0,2),(2,2),(4,3),(0,1)))
l = np.array((1,1,1,2,3,4,1,1,2,2,2,3,2,3,2,3,4))

# # pairwise
# x = x[:,np.newaxis,:]
# dist = np.add.reduce((x**2 + y**2 - 2*x*y), axis = -1)**0.5
dist = np.sqrt(np.sum(np.square(x[:,np.newaxis,:] - y), axis = -1))
# print(dist)

points = np.concatenate((x,y))
# print(points)

# # init
k = 2
centers = np.random.choice(points.shape[0], k, replace=False)
centers = points[centers]
print(centers)

