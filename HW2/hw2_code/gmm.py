import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """

        prob = np.exp(logit - np.max(logit, axis=-1, keepdims=True)) / np.sum(np.exp(logit - np.max(logit, axis=-1, keepdims=True)), axis=-1, keepdims=True)
        return prob
    
    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """

        s = np.log(np.sum(np.exp(logit - np.max(logit, axis=-1, keepdims=True)), axis=-1, keepdims=True)) + np.max(logit, axis=-1, keepdims=True)
        return s

    # # for undergraduate student
    # def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
    #     """
    #     Args:
    #         points: N x D numpy array
    #         mu_i: (D,) numpy array, the center for the ith gaussian.
    #         sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
    #     Return:
    #         pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

    #     Hint:
    #         np.diagonal() should be handy.
    #     """

    #     raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        D = mu_i.shape[0]
        try:
            inv = np.linalg.inv(sigma_i)
        except np.linalg.LinAlgError:
            inv = np.linalg.inv(sigma_i + SIGMA_CONST)

        semi = (points - mu_i) @ inv
        NN =  np.sum((-0.5 * (semi.T * (points - mu_i).T)), axis=0)
        # NN =  -0.5 * (semi @ (semi.T * (points - mu_i).T))
        normal_pdf = (1/((2*np.pi) ** (D/2))) * (np.linalg.det(sigma_i) ** (-0.5)) * np.exp(NN)
        return normal_pdf



    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed


        pi = np.ones((self.K,)) / self.K
        mu = np.ones((self.K, self.D))
        for i in range(self.K):
            mu[i] = self.points[int(np.random.uniform(0, self.N-1))]
        # mu = np.eye(3)
        sigma = np.ones((self.K, self.D, self.D)) * np.eye(self.D)

        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        if full_matrix is True:

            ll = np.ones((self.points.shape[0], self.K))
            for i in range(self.points.shape[0]):
                for k in range(self.K):
                    pdf = self.multinormalPDF(self.points[i], mu[k], sigma[k])
                    ll[i, k] = np.log( pi[k] + LOG_CONST ) + np.log( pdf + LOG_CONST )
            
            # print(ll)
            # print(ll.shape)

        return ll

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        if full_matrix is True:
            gamma = self.softmax(self._ll_joint(pi, mu, sigma, full_matrix=FULL_MATRIX))

        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        if full_matrix is True:
            N_k = gamma.sum(axis=0)

            pi_new = N_k / self.points.shape[0]
            # print(pi_new)

            mu_new = ( (gamma.T @ self.points).T / N_k ).T
            # print(mu_new)

            diff = np.ones((self.K, self.points.shape[0], self.points.shape[1]))
            sigma_new = np.ones((self.K, self.points.shape[1], self.points.shape[1]))
            for i in range(self.K):
                diff = self.points - mu_new[i]
                sigma_new[i] = ( np.dot(gamma[:,i].T * diff.T, diff) ) / N_k[i]
            # print(sigma_new)
        
        return pi_new, mu_new, sigma_new

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

