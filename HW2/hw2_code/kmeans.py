
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        centers = np.random.choice(self.points.shape[0], self.K, replace=False)
        self.centers = self.points[centers]
        return self.centers

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """

        # # # centers = np.array([self.points[np.random.choice(self.points.shape[0], 1)]]) # Initializes 1 center at random
        # # # dists = [] # in numpy standards
        # # # for i in range(self.K):
        # # #     for j in len(self.points):
        # # #         dists += pairwise_dist(centers[i], self.points[j])
        # # #         c = self.points[np.argmax(dists)] # EXCEPT THE ONES ALREADY DRAWN!!!

        return self.centers

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """
        
        d = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(d, axis=-1)
        return self.assignments

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        for i in range(self.K):
            cluster_curr = self.points[np.where(self.assignments == i)[0]] # Assume all clusters have at least one point assigned
            self.centers[i] = np.mean(cluster_curr, axis=0)
            
        return self.centers

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """

        l = np.zeros(self.K)
        for i in range(self.K):
            l[i] = np.sum(np.square(self.points[np.where(self.assignments == i)[0]] - self.centers[i]))

        self.loss = np.sum(l)

        return self.loss

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        
        # print("tolerance: ", self.rel_tol)
        # print("max iterations: ", self.max_iters)
        prev_loss = 1e7

        for i in range(self.max_iters):
            self.assignments = self.update_assignment()
            self.centers = self.update_centers()
            for j in range(self.K):
                if len(np.where(self.assignments == j)[0]) == 0:
                    c = np.random.choice(self.points.shape[0])
                    self.centers[j] = self.points[c]
                    self.assignments = self.update_assignment()

            loss = self.get_loss()
            # print("Loss in iteration ", i, " is: ", loss)
            dl = (abs(loss - prev_loss)/prev_loss)
            # print("dl = ", dl)

            if dl <= self.rel_tol:
                return self.centers, self.assignments, self.loss
            else:
                prev_loss = loss
                # print(i, " iterations done")

        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """

        dist = np.sqrt(abs(np.sum(np.square(x), axis=-1, keepdims=True) + np.sum(np.square(y), axis=-1) - 2*(np.dot(x, y.T))))
        return dist
