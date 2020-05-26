"""Radial Basis Function Networks"""

# Christoph Schröder <cs@keksdev.de> 2019

import numpy as np
from scipy.spatial import distance
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


class Rbfn(BaseEstimator):
    """Radial Basis Function Networks

    Based on George, Anjith, and Aurobinda Routray. “A Score Level Fusion Method for Eye Movement Biometrics.”
    Pattern Recognition Letters 82 (2016): 207–15. https://doi.org/10.1016/j.patrec.2015.11.020.

    Implementation for the paper:
    Schröder, Christoph, Sahar Mahdie Klim Al Zaidawi, Martin H.U. Prinzler, Sebastian Maneth, and Gabriel Zachmann.
    ‘Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length’.
    In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, 1–7. CHI ’20.
    Honolulu, HI, USA: Association for Computing Machinery, 2020. https://doi.org/10.1145/3313831.3376534.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form.
    use_cuda : bool, default: False
        Calculate the pseudo inverse with cude. This currently is not working as the result is wrong.
    """
    def __init__(self, n_clusters=8, use_cuda=False, random_state=42):
        # hyperparameter
        self.random_state = random_state
        self.n_clusters = n_clusters

        # learned parameters
        self.w = np.array([])
        self.mus = np.array([])
        self.betas = np.array([])

        self.classes_ = []

        # we need scaled data and classes one-hot encoded
        self.enc = OneHotEncoder(categories='auto', sparse=False)

        # config
        self.use_cuda = use_cuda

    def fit(self, X, y):
        """ Train the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")


        # allocate space for our parameters
        self.mus = np.zeros((self.n_clusters * n_classes, X.shape[1]))
        self.betas = np.zeros(len(self.mus))

        # 1. find n_clusters support vectors
        # calculates all \mu s with k-means:
        #   - 1 nearest neighbor
        #   - euclidean distance
        #   - max 100 iterations
        mu_count = 0
        # estimate \mu and \beta for each of the self.n_clusters * n_classes neurons
        for i, c in enumerate(self.classes_):
            # range to save our parameters in
            param_slice = slice(i * self.n_clusters, (i + 1) * self.n_clusters)

            # subset with only training data for the current class
            X_subset = X[y==c, :]

            # mus for the current class
            kmns = KMeans(n_clusters=self.n_clusters, max_iter=100, random_state=self.random_state)
            kmns.fit(X_subset)
            self.mus[param_slice] = kmns.cluster_centers_

            # 2. estimate n_clusters \betas
            # \beta = 1/(2*\sigma^2)
            # \sigma = mean distance between cluster center and all of it's elements
            for j, c in enumerate(kmns.cluster_centers_):
                cluster_samples = X_subset[kmns.labels_ == j]
                sigma = np.mean(distance.cdist(cluster_samples, np.asarray([c])))
                # add small delta to avoid division by zero
                self.betas[mu_count] = 1 / (2 * (sigma+0.00001)**2)
                mu_count += 1

        # 3. find n_classes weigths (why not n_classes * self.n_clusters?)
        # input all data and find w using the Moore–Penrose pseudoinverse
        # A = f_{i,j}(x_k)
        # i = 1,...,self.n_clusters (K)
        # j = 1,...,n_classes (m)
        # k = 1,...,n_samples (n = m*c)
        # Aŵ=ŷ
        A = self._activate(X)

        # if we divide by 0 in the beta calculation we get nans, replace them by 0
        nan_count = np.sum(np.isnan(A))
        if nan_count > 0:
            print("replacing zeros in activation".format(nan_count))
            A[np.isnan(A)] = 0

        if self.use_cuda:
            print("Calculating pinv of size {}".format(A.shape))
            print("WARNING: CUDA currently gives the wrong result!")
            print("Using cuda")
            import pycuda.gpuarray as gpuarray
            import skcuda.linalg as linalg
            linalg.init()
            print("Moving A to GPU")
            a_gpu = gpuarray.to_gpu(np.asarray(A.T, np.float32))
            print("Calculating pinv")
            a_inv_gpu = linalg.pinv(a_gpu)
            print("Moving A back to CPU")
            Ap = a_inv_gpu.get().T
        else:
            #Ap = scipy.linalg.pinv(A)
            Ap = np.linalg.pinv(A)

        # y in one hot encoding
        y_hot = self.enc.fit_transform(y.reshape(-1, 1))
        self.w = Ap @ y_hot

        return self

    def predict(self, X):
        """Predict the class each sample in X belongs to.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Class each sample belongs to.
        """

        return self.enc.inverse_transform(self.predict_proba(X))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Pseudo probability.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        activation = self._activate(X)
        prediction = activation @ self.w

        # normalize to values between 0 and 1
        prediction_normalized = softmax(prediction, axis=1)
        return prediction_normalized

    def _activate(self, X):
        """ Calculate activation vector for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        :return: ndarray
            array, shape [n_samples, n_classes * n_clusters]
        """
        activations = []
        for f in X:
            # Eq. 3 from the paper:
            # f is broadcasted to len(mus)
            d = f - self.mus
            # squared norm
            res = np.einsum('ij,ij->i',d,d)
            activations.append(np.exp(-self.betas * res))

        return np.asarray(activations)
