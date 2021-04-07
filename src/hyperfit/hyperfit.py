import zeus
import emcee
import numpy as np
from scipy.special import loggamma
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

        self.mcmc_samples = None
        self.mcmc_lnlike = None

    # Code to compute normal vectors from cartesian coordinates
    def compute_normal(self, coords=None, vert_scat=None):

        if coords is None:
            coords = self.coords

        if vert_scat is None:
            vert_scat = self.vert_scat

        alpha = coords
        alpha[self.vertaxis] = -1
        beta = coords[self.vertaxis]
        normalpha = np.sum(alpha ** 2)
        unitalpha = alpha / normalpha ** 2
        normal = -unitalpha * beta
        norm_scat = np.fabs(vert_scat / np.sqrt(normalpha))

        return normal, norm_scat

    # Code to compute cartesian coordinates from normal vectors
    def compute_cartesian(self, normal=None, norm_scat=None):

        if normal is None:
            normal = self.normal

        if norm_scat is None:
            norm_scat = self.norm_scat

        nTn = np.sum(normal ** 2, axis=0)
        coords = -normal / normal[self.vertaxis]
        coords[self.vertaxis] = nTn / normal[self.vertaxis]
        vert_scat = np.fabs(norm_scat * np.sqrt(nTn) / np.fabs(normal[self.vertaxis]))

        return coords, vert_scat

    def bessel_cochran(self, sigma):
        sigma_dbar = (
            np.sqrt(0.5 * self.ndata)
            * np.exp(loggamma(0.5 * (self.ndata - self.ndims)) - loggamma(0.5 * (self.ndata - self.ndims + 1.0)))
        ) * sigma
        return sigma_dbar

    # The posterior function
    def lnpost(self, params):
        if params.ndim == 1:
            params = params.reshape(-1, len(params))
        lnpost = np.where(params[:, -1] > 0.0, np.sum(self.weights * self.lnlike(params), axis=-1), -np.inf)
        return lnpost

    # The log-likelihood function for each data point
    def lnlike(self, params):

        if params.ndim == 1:
            params = params.reshape(-1, len(params))

        nTn = np.sum(params[:, :-1] ** 2, axis=1)
        nTcn = np.einsum("ki,dij,kj->dk", params[:, :-1], self.cov, params[:, :-1])
        orthvariance = params[:, -1] ** 2 + nTcn / nTn
        originoffset = np.einsum("ki,di->dk", params[:, :-1], self.data) / np.sqrt(nTn) - np.sqrt(nTn)
        lnlike = -0.5 * (np.log(orthvariance) + (originoffset ** 2) / orthvariance)

        return lnlike.T

    # A routine to optimize the model given some data
    def optimize(self, begin, verbose=False):

        if len(begin) != self.ndims + 1:
            raise ValueError("length of begin not equal to N dimensions + 1")

        begin = np.array(begin)
        self.normal, self.norm_scat = self.compute_normal(coords=begin[:-1], vert_scat=begin[-1])
        result = basinhopping(
            lambda *args: -self.lnpost(*args),
            np.concatenate([self.normal, [self.norm_scat]]),
            niter=10,
            minimizer_kwargs={"method": "Nelder-Mead", "tol": 1.0e-6},
        )
        if verbose:
            print(result)
        self.normal = result["x"][:-1]
        self.norm_scat = np.fabs(result["x"][-1])
        self.norm_scat = self.bessel_cochran(self.norm_scat)
        self.coords, self.vert_scat = self.compute_cartesian()

        return self

    # A routine run a zeus MCMC on the model given the data
    def zeus(self, begin, max_iter=100000, batchsize=1000, verbose=False):

        if len(begin) != self.ndims + 1:
            raise ValueError("length of begin not equal to N dimensions + 1")

        # Set up emcee. Start the walkers in a small 1 percent ball around the best fit
        self.optimize(begin)
        nwalkers = 4 * (self.ndims + 1)
        begin = [
            [(0.01 * np.random.rand() + 0.995) * j for j in np.concatenate([self.normal, [self.norm_scat]])]
            for _ in range(nwalkers)
        ]

        sampler = zeus.EnsembleSampler(nwalkers, self.ndims + 1, self.lnpost, vectorize=True, verbose=False)

        old_tau = np.inf
        niter = 0
        converged = 0
        while ~converged:
            sampler.run_mcmc(begin, nsteps=batchsize)
            tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
            converged = np.all(10.0 * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            old_tau = tau
            begin = None
            niter += 1000
            if verbose:
                print("Niterations/Max Iterations: ", niter, "/", max_iter)
                print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(10.0 * tau))
            if niter >= max_iter:
                break

        # Remove burn-in and and save the samples
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T
        self.mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1, :], norm_scat=samples[-1, :])).T
        self.mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return self

    # A routine run a zeus MCMC on the model given the data
    def emcee(self, begin, max_iter=100000, batchsize=1000, verbose=False):

        if len(begin) != self.ndims + 1:
            raise ValueError("length of begin not equal to N dimensions + 1")

        # Set up emcee. Start the walkers in a small 1 percent ball around the best fit
        self.optimize(begin)
        nwalkers = 4 * (self.ndims + 1)
        begin = [
            [(0.01 * np.random.rand() + 0.995) * j for j in np.concatenate([self.normal, [self.norm_scat]])]
            for _ in range(nwalkers)
        ]

        sampler = emcee.EnsembleSampler(nwalkers, self.ndims + 1, self.lnpost, vectorize=True)

        old_tau = np.inf
        niter = 0
        converged = 0
        while ~converged:
            sampler.run_mcmc(begin, nsteps=batchsize, progress=verbose)
            tau = sampler.get_autocorr_time(discard=int(0.5 * niter), tol=0)
            converged = np.all(50.0 * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            old_tau = tau
            begin = None
            niter += 1000
            if verbose:
                print("Niterations/Max Iterations: ", niter, "/", max_iter)
                print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(50.0 * tau))
            if niter >= max_iter:
                break

        # Remove burn-in and and save the samples
        tau = sampler.get_autocorr_time(discard=int(0.5 * niter), tol=0)
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T
        self.mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1, :], norm_scat=samples[-1, :])).T
        self.mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return self


if __name__ == "__main__":

    # Run some tests, copying the examples in Robotham and Obreschkow 2015
    from src.hyperfit.hyperfit import HyperFit
    from src.hyperfit.data import GAMAsmVsize, Hogg, TFR, FP6dFGS, MJB

    data = GAMAsmVsize()
    hf = HyperFit(data.xs, data.cov, weights=data.weights).zeus([1.0, 0.0, 0.0], verbose=False)
    print(hf.coords, hf.vert_scat, np.mean(hf.mcmc_samples, axis=0), np.std(hf.mcmc_samples, axis=0))
    hf = HyperFit(data.xs, data.cov, weights=data.weights).emcee([1.0, 0.0, 0.0], verbose=False)
    print(hf.coords, hf.vert_scat, np.mean(hf.mcmc_samples, axis=0), np.std(hf.mcmc_samples, axis=0))
