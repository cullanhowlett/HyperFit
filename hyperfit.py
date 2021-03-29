import numpy as np
import pandas as pd
from scipy.optimize import basinhopping


class HyperFit(object):

    # Initialise the necessary parameters and perform checks
    def __init__(self, data, cov, weights=None, vertaxis=-1):

        self.ndata = np.shape(data)[0]
        self.ndims = np.shape(data)[1]
        self.data = data
        self.cov = cov
        self.weights = np.ones(len(data)) if weights is None else weights
        self.vertaxis = vertaxis

        # Some variables to store the two sets of fitting coordinates and scatter parameters
        self.coords = np.empty(self.ndims)
        self.normal = np.empty(self.ndims)
        self.vert_scat = 0.0
        self.norm_scat = 0.0

    # Code to compute normal vectors from cartesian coordinates
    def compute_normal(self):

        alpha = self.coords
        alpha[self.vertaxis] = -1
        beta = self.coords[self.vertaxis]
        normalpha = np.sum(alpha ** 2)
        unitalpha = alpha / normalpha ** 2
        self.normal = -unitalpha * beta
        self.norm_scat = self.vert_scat / np.sqrt(normalpha)

        return self

    # Code to compute cartesian coordinates from normal vectors
    def compute_cartesian(self):

        nTn = np.sum(self.normal ** 2)
        self.coords = -self.normal / self.normal[self.vertaxis]
        self.coords[self.vertaxis] = nTn / self.normal[self.vertaxis]
        self.vert_scat = self.norm_scat * np.sqrt(nTn) / np.fabs(self.normal[self.vertaxis])

        return self

    # The posterior function
    def lnpost(self, params):
        lnpost = np.sum(self.weights * self.lnlike(params))
        return lnpost

    # The log-likelihood function for each data point
    def lnlike(self, params):

        nTn = np.sum(params[:-1] ** 2)
        nTcn = np.inner(params[:-1], np.inner(self.cov, params[:-1]))
        orthvariance = params[-1] ** 2 + nTcn / nTn
        originoffset = np.inner(self.data, params[:-1]) / np.sqrt(nTn) - np.sqrt(nTn)
        lnlike = -0.5 * (np.log(orthvariance) + (originoffset ** 2) / orthvariance)

        return lnlike

    # A routine to optimize the model given some data
    def optimize(self, begin):

        begin = np.array(begin)
        self.coords = begin[:-1]
        self.vert_scat = begin[-1]
        self.compute_normal()
        result = basinhopping(
            lambda *args: -self.lnpost(*args),
            np.concatenate([self.normal, [self.norm_scat]]),
            niter=10,
            minimizer_kwargs={"method": "Nelder-Mead", "tol": 1.0e-6},
        )
        self.normal = result["x"][:-1]
        self.norm_scat = result["x"][-1]
        self.compute_cartesian()

        return self


if __name__ == "__main__":

    # Run a little 2D test, copying the 2D Mass-size relation in 5.1 of Robotham and Obreschkow 2015
    from hyperfit import HyperFit

    GAMA = pd.read_csv("test_data/GAMAsmVsize.txt", delim_whitespace=True, escapechar="#")

    data = np.array([GAMA["logmstar"].to_numpy(), GAMA["logrekpc"].to_numpy()]).T
    cov = np.array(
        [
            [GAMA["logmstar_err"].to_numpy() ** 2, np.zeros(len(GAMA))],
            [np.zeros(len(GAMA)), GAMA["logrekpc_err"].to_numpy() ** 2],
        ]
    ).T
    hp = HyperFit(data, cov, weights=GAMA["weights"].to_numpy()).optimize([1.0, 0.0, 0.0])

    print(hp.coords, hp.norm_scat, hp.vert_scat)
