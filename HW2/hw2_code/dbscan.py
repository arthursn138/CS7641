import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """

        cluster_idx = -np.ones(self.dataset.shape[0]) # Initializes all points as unvisited/noise
        visitedIndices = set()
        C = 0   # Initialize the first cluster as 0
        for i in range(self.dataset.shape[0]):
            index = i
            # print("i = ", i)
            if np.size(np.where(np.array(list(visitedIndices)) == i)) == 0:
                visitedIndices.add(i)
                # print("visited: ", visitedIndices)
                neighborPts = self.regionQuery(i)
                # print("neighbors of ", i, " : ", neighborPts)
                if np.size(neighborPts) >= self.minPts:
                    # print("current cluster: ", C)
                    self.expandCluster(index, neighborPts, C, cluster_idx, visitedIndices)
                    C += 1

        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """

        cond = True
        while cond:
            for j in neighborIndices:
                if np.size(np.where(np.sort(np.array(list(visitedIndices))) == j)) == 0:
                    visitedIndices.add(j)
                    neighborsPrime = self.regionQuery(j)
                    if neighborsPrime.size >= self.minPts:
                        neighborIndices = np.unique(np.sort(np.concatenate((neighborIndices, neighborsPrime), axis=None)))

                if cluster_idx[j] == -1:
                    cluster_idx[j] = C

            if np.array_equal(neighborIndices, np.sort(np.array(list(visitedIndices)))):
                cond = False
        
            elif np.array_equal(self.dataset.shape[0], np.shape(np.array(list(visitedIndices)))[0]):
                cond = False

        return

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        
        indices = np.ndarray.flatten(np.argwhere(pairwise_dist(self.dataset[pointIndex], self.dataset) <= self.eps))
        return indices
            