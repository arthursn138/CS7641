import numpy as np
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        
        # Centering
        centered = X - X.mean(axis=0)
        U, S, V = np.linalg.svd(centered, full_matrices=False)
        # Clipping (notebook says D < N)
        _, D = np.shape(X)
        self.U = U[:, :D]
        self.S = S[:D]
        self.V = (V[:D, :D])

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        
        self.fit(data)
        centered = data - data.mean(axis=0)
        X_new = np.dot(centered, (self.V[:K, :]).T)
        return X_new

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        
        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        
        self.fit(data)
        centered = data - data.mean(axis=0)
        cumm_var = 0
        K = 0
        for i in range(len(self.S)):
            cumm_var += self.S[i]/np.sum(self.S)
            if cumm_var >= retained_variance:
                K = i

        if K == 0:
            K = np.shape(self.S)[0] - 1

        X_new = np.dot(centered, (self.V[:K,:]).T)
        return X_new

    def get_V(self) -> np.ndarray:
        """ Getter function for value of V """

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig=None) -> None:  # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        Hint: Remember to label each of the plots when looping through. Refer to https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """

        # We know that numpy's svd func returns the singular values sorted in descending order
        X_new = self.transform(X, K=2)
        print(f"Data shape before PCA: {X.shape}")
        print(f"Data shape before PCA: {X_new.shape}")
        print(f"Labels: {np.unique(y)}")
        
        for i in range(X_new.shape[0]):
            if y[i] == 0:
                color = 'blue'
            elif y[i] == 1:
                color = 'magenta'
            elif y[i] == 2:
                color = 'red'

            ax = plt.scatter(X_new[i,0], X_new[i,1], c=color, marker="x")
        
        # list = [str(np.unique(y)[0]), str(np.unique(y)[1]), str(np.unique(y)[2])]
        # plt.legend(handles=ax.legend_elements()[0], labels=list)

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()
