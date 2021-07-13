import copy
import zeus
import emcee
import numpy as np
from scipy.special import loggamma
from scipy.optimize import differential_evolution


class LinFit(object):

    """The LinFit class.

    Implements methods to fit straight lines or planes, including taking data and a covariance matrix and fitting
    either to just find the best fit, or run an MCMC. Has four main attributes, that are useful for
    accessing other information after running 'optimize', 'emcee' or 'zeus'.

    Attributes
    ----------
    coords: ndarray
        N dimensional array holding the best-fitting HyperFit parameters in the data coordinates
        after a call to 'optimize' or one of the MCMC routines. Otherwise zeros.
    normal: ndarray
        N dimensional array holding the best-fitting HyperFit parameters in the normal unit vectors
        after a call to 'optimize' or one of the MCMC routines. Otherwise zeros.
    vert_scat: float
        Holds the best-fitting scatter in the vertical axis of the data coordinates, after a call to 'optimize'
        or one of the MCMC routines. Otherwise zero.
    norm_scat: float
        Holds the best-fitting scatter normal to the plane, after a call to 'optimize'
        or one of the MCMC routines. Otherwise zero.
    normal_bounds: sequence
        Holds the prior bounds in the normal unit vectors, after bounds in the data coordinates have
        been passed to a call to 'optimize' or one of the MCMC routines. Otherwise None.

    Args
    ----
    data: ndarray
        The N x D dimensional data vector
    cov: ndarray
        The N x N x D dimensional set of covariance matrices.
    weights: ndarray, optional
        D dimensional array of weights for each data. Default is None, in which can unit weights are assumed
        for each data point.
    vertaxis: float, optional
        Specifies which of the coordinate axis is to be treated as the 'vertical' axis (i.e,. 'y' for 2D data).
        Default is -1, in which case the last axis will be treated as vertical.

    """

    # Initialise the necessary parameters and perform checks
    def __init__(self, data, cov, weights=None, vertaxis=-1):

        self.ndims = np.shape(data)[0]
        self.ndata = np.shape(data)[1]
        self.data = data
        self.cov = cov
        self.weights = np.ones(self.ndata) if weights is None else weights
        self.vertaxis = vertaxis

        # Some variables to store the two sets of fitting coordinates and scatter parameters
        self.coords = np.zeros(self.ndims)
        self.normal = np.zeros(self.ndims)
        self.vert_scat = 0.0
        self.norm_scat = 0.0
        self.normal_bounds = None

    # Code to compute normal vectors from cartesian coordinates
    def compute_normal(self, coords=None, vert_scat=None):

        """Converts from data coordinates to the normal vector.

        Args
        ----
            coords : ndarray, optional
                N x M dimensional array of coordinates. Default is None, which means use the values
                stored in the self.coords attribute.
            vert_scat : ndarray, optional
                M dimensional array of scatter values. Default is None, which means use the values
                stored in the self.vert_scat attribute.

        Returns
        -------
            normal : ndarray
                N x M dimensional array of normal unit vectors.
            norm_scat : ndarray
                M dimensional array of scatters normal to the N-1 dimensional plane.

        """

        if coords is None:
            coords = self.coords.reshape(self.ndims, -1)

        if vert_scat is None:
            vert_scat = self.vert_scat

        alpha = copy.copy(coords)
        alpha[self.vertaxis] = -1
        beta = coords[self.vertaxis]
        normalpha = np.sum(alpha ** 2, axis=0)
        unitalpha = alpha / normalpha
        normal = -unitalpha * beta
        norm_scat = np.fabs(vert_scat / np.sqrt(normalpha))

        # print(coords, vert_scat, normal, norm_scat)

        return normal, norm_scat

    # Code to compute cartesian coordinates from normal vectors
    def compute_cartesian(self, normal=None, norm_scat=None):

        """Converts from the normal vector to the data coordinates.

        Args
        ----
            normal : ndarray, optional
                N x M dimensional array of unit vectors. Default is None, which means use the values
                stored in the self.normal attribute.
            norm_scat : ndarray, optional
                M dimensional array of scatter values normal to the plane. Default is None, which means
                use the values stored in the self.norm_scat attribute.

        Returns
        -------
            coords : float
                N x M dimensional array of points in the data coordinates.
            vert_scat: float
                M dimensional array of scatters along the vertical axis of the data.

        """

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

        """Corrects the sample scatter to the population scatter using the Bessel and Cochran corrections.

        The intrinsic scatter fit from the likelihood is generally not equal to the underlying population scatter
        This is 1) because the standard deviation is estimated from a finite number of data samples, and 2) because
        the maximum likelihood value of the variance is not the maximum likelihood value of the standard deviation.
        These are corrected by the so-called Bessel and Cochran corrections respectively. This function applies these
        corrections based on the number of data points and dimensionality of the fitted plane.

        Args
        ----
            sigma : ndarray
                M dimensional array of scatter values.

        Return
        ------
            sigma_corr : ndarray
                M dimensional array of corrected scatter values.

        """

        sigma_corr = (
            np.sqrt(0.5 * self.ndata)
            * np.exp(loggamma(0.5 * (self.ndata - self.ndims)) - loggamma(0.5 * (self.ndata - self.ndims + 1.0)))
        ) * sigma

        return sigma_corr

    # The posterior function
    def _lnpost(self, params):

        if params.ndim == 1:
            params = params.reshape(-1, len(params))

        lnpost = np.sum(self.weights * self._lnlike(params), axis=-1) + self._lnprior(params)

        return lnpost

    # The flat prior for the fit
    def _lnprior(self, params):

        lnprior = np.zeros(len(params))
        for i, (param, bounds) in enumerate(zip(params.T, self.normal_bounds)):
            lnprior += np.where(np.logical_or(param < bounds[0], param > bounds[1]), -np.inf, 0.0)

        return lnprior

    # The log-likelihood function for each data point
    def _lnlike(self, params):

        if params.ndim == 1:
            params = params.reshape(-1, len(params))

        nTn = np.sum(params[:, :-1] ** 2, axis=1)
        nTcn = np.einsum("ki,ijd,kj->dk", params[:, :-1], self.cov, params[:, :-1])
        orthvariance = params[:, -1] ** 2 + np.where(nTn > 0, nTcn / nTn, 0)
        originoffset = np.where(
            nTn > 0, np.einsum("ki,id->dk", params[:, :-1], self.data) / np.sqrt(nTn) - np.sqrt(nTn), 0.0
        )
        lnlike = -0.5 * (np.log(orthvariance) + (originoffset ** 2) / orthvariance)

        return lnlike.T

    # Convert a sequence of bounds in data coordinates to bounds on the normal vector by optimizing for the maximum
    # and minimum values of each normal vector coordinate across the original parameter space.
    def _convert_bounds(self, bounds):

        normal_bounds = []
        for i in range(len(bounds) - 1):
            new_min = differential_evolution(
                lambda x: self.compute_normal(coords=x[:-1], vert_scat=x[-1])[0][i], bounds, tol=1.0e-6
            )["fun"]
            new_max = -differential_evolution(
                lambda x: -self.compute_normal(coords=x[:-1], vert_scat=x[-1])[0][i], bounds, tol=1.0e-6
            )["fun"]
            normal_bounds.append((new_min, new_max))
        new_min = differential_evolution(
            lambda x: self.compute_normal(coords=x[:-1], vert_scat=x[-1])[1], bounds, tol=1.0e-6
        )["fun"]
        new_max = -differential_evolution(
            lambda x: -self.compute_normal(coords=x[:-1], vert_scat=x[-1])[1], bounds, tol=1.0e-6
        )["fun"]
        normal_bounds.append((new_min, new_max))
        normal_bounds = tuple(normal_bounds)

        return normal_bounds

    def get_sigmas(self, normal=None, norm_scat=None):

        """Calculates the offset between each data point and a plane in
           units of the standard deviation, i.e., in terms of x-sigma.

        Args
        ----
            normal : ndarray, optional
                N x M dimensional array of unit vectors. Default is None, which means use the values
                stored in the self.normal attribute.
            norm_scat : ndarray, optional
                M dimensional array of scatter values normal to the plane. Default is None, which means
                use the values stored in the self.norm_scat attribute.

        Returns
        -------
            sigmas: ndarray
                D x M dimensional array containing the offsets of the D data points, in units of the
                standard deviation from the M models.

        """

        if normal is None:
            normal = self.normal.reshape(self.ndims, -1)

        if norm_scat is None:
            norm_scat = self.norm_scat

        nTn = np.sum(normal ** 2, axis=0)
        nTcn = np.einsum("ik,ijd,jk->dk", normal, self.cov, normal)
        orthvariance = norm_scat ** 2 + np.where(nTn > 0, nTcn / nTn, 0)
        originoffset = np.where(nTn > 0, np.einsum("ik,id->dk", normal, self.data) / np.sqrt(nTn) - np.sqrt(nTn), 0.0)

        return np.sqrt((originoffset ** 2) / orthvariance)

    def optimize(self, bounds, tol=1.0e-6, verbose=False):

        """Find the best-fitting line/plane/hyperplane.

        Fits the N x D dimensional self.data using scipy.optimise's basinhopping + Nelder-Mead algorithm. Pretty robust.

        Args
        ----
            bounds : sequence
                Bounds for variables. Must be a set of N + 1 (min, max) pairs, one for each free parameter,
                defining the finite lower and upper bounds. Passed straight through to scipy.differential_evolution
            tol: float, optional
                The optimisation tolerance.
            verbose : bool, optional
                If True prints out the full dictionary returned by scipy.optimize.basinhopping.

        Return
        ------
            coords : ndarray
                N dimensional array containing the best-fitting parameters.
            vert_scat: float
                The scatter in the vertical axis, corrected using the Bessel-Cochran correction.
            log_posterior: float
                The log posterior at the best-fitting parameters.

        Raises
        ------
            ValueError: If the number of pairs in 'bounds' is not equal to N + 1.

        Note
        ----
        If you want to access the best-fitting parameters in the normal coordinates and the scatter normal to the plane,
        these are stored in the self.normal and self.norm_scat class attributes respectively following a call to optimize.

        """

        if len(bounds) != self.ndims + 1:
            raise ValueError("Number of bounds (min, max) pairs not equal to N dimensions + 1")

        self.normal_bounds = self._convert_bounds(bounds)

        result = differential_evolution(lambda *args: -self._lnpost(*args), self.normal_bounds, tol=tol)

        if verbose:
            print(result)

        self.normal = result["x"][:-1]
        self.norm_scat = np.fabs(result["x"][-1])
        self.norm_scat = self.bessel_cochran(self.norm_scat)
        self.coords, self.vert_scat = self.compute_cartesian()

        return self.coords, self.vert_scat, -result["fun"][0]

    # A routine run a zeus MCMC on the model given the data
    def zeus(self, bounds, max_iter=100000, batchsize=1000, ntau=10.0, tautol=0.05, verbose=False):

        """Run an MCMC on the data using the zeus sampler (Karamanis and Beutler 2020).

        The MCMC runs in batches, checking convergence at the end of each batch until either the chain is well converged
        or the maximum number of iterations has been reached. Convergence is defined as the point when the chain is longer
        than ntau autocorrelation lengths, and the estimate of the autocorrelation length varies less than tautol between batches.
        Burn-in is then removed from the samples, before they are flattened and returned.

        Args
        ----
            bounds : sequence
                Bounds for variables. Must be a set of N + 1 (min, max) pairs, one for each free parameter,
                defining the finite lower and upper bounds. Passed straight through to scipy.differential_evolution, and
                used to set the prior for the MCMC sampler.
            max_iter: int, optional
                The maximum number of MCMC iterations.
            batchsize : int, optional
                The size of each batch, between which we check convergence.
            ntau: float, optional
                The minimum number of autocorrelation lengths to require before convergence.
            tautol: float, optional
                The maximum fractional deviation between successive values of the autocorrelation length required for convergence.
            verbose: bool, optional
                Whether or not to print out convergence statistics and progress.


        Return
        ------
            mcmc_samples : ndarray
                (N + 1) x Nsamples dimensional array containing the flattened, burnt-in MCMC samples. First N dimensions
                are the parameters of the plane. Last dimension is intrinsic scatter in the vertical axis.
            mcmc_lnlike : ndarray
                Nsamples dimensional array containing the log-likelihood for each MCMC sample.

        Raises
        ------
            ValueError: If the number of values in 'begin' is not equal to N + 1.

        Note
        ----
        Also calls 'optimize' and stores the results in the relevant class attributes if you want to access the best-fit.

        """

        if len(bounds) != self.ndims + 1:
            raise ValueError("Number of bounds (min, max) pairs not equal to N dimensions + 1")

        # Set up Zeus. Start the walkers in a small 1 percent ball around the best fit
        self.optimize(bounds)
        nwalkers = 4 * (self.ndims + 1)
        begin = [
            [(0.01 * np.random.rand() + 0.995) * j for j in np.concatenate([self.normal, [self.norm_scat]])]
            for _ in range(nwalkers)
        ]

        sampler = zeus.EnsembleSampler(nwalkers, self.ndims + 1, self._lnpost, vectorize=True, verbose=False)

        old_tau = np.inf
        niter = 0
        converged = 0
        while ~converged:
            sampler.run_mcmc(begin, nsteps=batchsize)
            tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
            converged = np.all(ntau * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < tautol)
            old_tau = tau
            begin = None
            niter += 1000
            if verbose:
                print("Niterations/Max Iterations: ", niter, "/", max_iter)
                print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(ntau * tau))
            if niter >= max_iter:
                break

        # Remove burn-in and and save the samples
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T
        mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1, :], norm_scat=samples[-1, :]))
        mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return mcmc_samples, mcmc_lnlike

    # A routine run a zeus MCMC on the model given the data
    def emcee(self, bounds, max_iter=100000, batchsize=1000, ntau=50.0, tautol=0.05, verbose=False):

        """Run an MCMC on the data using the emcee sampler (Foreman-Mackay et. al., 2013).

        The MCMC runs in batches, checking convergence at the end of each batch until either the chain is well converged
        or the maximum number of iterations has been reached. Convergence is defined as the point when the chain is longer
        than ntau autocorrelation lengths, and the estimate of the autocorrelation length varies less than tautol between batches.
        Burn-in is then removed from the samples, before they are flattened and returned.

        Args
        ----
            bounds : sequence
                Bounds for variables. Must be a set of N + 1 (min, max) pairs, one for each free parameter,
                defining the finite lower and upper bounds. Passed straight through to scipy.differential_evolution, and
                used to set the prior for the MCMC sampler.
            max_iter: int, optional
                The maximum number of MCMC iterations.
            batchsize : int, optional
                The size of each batch, between which we check convergence.
            ntau: float, optional
                The minimum number of autocorrelation lengths to require before convergence.
            tautol: float, optional
                The maximum fractional deviation between successive values of the autocorrelation length required for convergence.
            verbose: bool, optional
                Whether or not to print out convergence statistics and progress.


        Return
        ------
            mcmc_samples : ndarray
                (N + 1) x Nsamples dimensional array containing the flattened, burnt-in MCMC samples. First N dimensions
                are the parameters of the plane. Last dimension is intrinsic scatter in the vertical axis.
            mcmc_lnlike : ndarray
                Nsamples dimensional array containing the log-likelihood for each MCMC sample.

        Raises
        ------
            ValueError: If the number of values in 'begin' is not equal to N + 1.

        Note
        ----
        Also calls 'optimize' and stores the results in the relevant class attributes if you want to access the best-fit.

        """

        if len(bounds) != self.ndims + 1:
            raise ValueError("Number of bounds (min, max) pairs not equal to N dimensions + 1")

        # Set up emcee. Start the walkers in a small 1 percent ball around the best fit
        self.optimize(bounds, verbose=verbose)
        nwalkers = 4 * (self.ndims + 1)
        begin = [
            [(0.01 * np.random.rand() + 0.995) * j for j in np.concatenate([self.normal, [self.norm_scat]])]
            for _ in range(nwalkers)
        ]

        sampler = emcee.EnsembleSampler(nwalkers, self.ndims + 1, self._lnpost, vectorize=True)

        old_tau = np.inf
        niter = 0
        converged = 0
        while ~converged:
            sampler.run_mcmc(begin, nsteps=batchsize, progress=verbose)
            tau = sampler.get_autocorr_time(discard=int(0.5 * niter), tol=0)
            converged = np.all(ntau * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < tautol)
            old_tau = tau
            begin = None
            niter += 1000
            if verbose:
                print("Niterations/Max Iterations: ", niter, "/", max_iter)
                print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(ntau * tau))
            if niter >= max_iter:
                break

        # Remove burn-in and and save the samples
        tau = sampler.get_autocorr_time(discard=int(0.5 * niter), tol=0)
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T
        mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1], norm_scat=samples[-1]))
        mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return mcmc_samples, mcmc_lnlike


if __name__ == "__main__":

    from src.hyperfit.linfit import LinFit
    from src.hyperfit.data import ExampleData, GAMAsmVsize, TFR, FP6dFGS, MJB

    # Load in the ExampleData
    data = ExampleData()
    print(np.shape(data.xs), np.shape(data.cov), np.shape(data.weights))
    hf = LinFit(data.xs, data.cov, weights=data.weights)

    # Run an MCMC
    bounds = ((-5.0, 5.0), (-10.0, 10.0), (1.0e-5, 5.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=False)
    print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))

    # Have a look at the normal coordinates
    normal_mcmc_samples = np.vstack(hf.compute_normal(coords=mcmc_samples[:-1], vert_scat=mcmc_samples[-1]))
    print(hf.normal, hf.norm_scat)
    print(np.mean(normal_mcmc_samples, axis=1), np.std(normal_mcmc_samples, axis=1))

    # Have a look at the sigma offsets
    sigmas = hf.get_sigmas()
    print(hf.get_sigmas(normal=normal_mcmc_samples[:-1], norm_scat=normal_mcmc_samples[-1]))
