import zeus
import emcee
import numpy as np
from scipy.special import loggamma
from scipy.optimize import basinhopping


class HyperFit(object):

    """The HyperFit class.

    Implements all the HyperFit methods, including taking data and a covariance matrix and fitting
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

    Args
    ----
    data: ndarray
        The N x D dimensional data vector
    cov: ndarray
        The N x N x D dimensional set of covariance matrices.

    Keyword Args
    ------------
    weights: ndarray, optional
        D dimensional array of weights for each data. Default is None, in which can unit weights are assumed
        for each data point.
    vertaxis: float, optional
        Specifies which of the coordinate axis is to be treated as the 'vertical' axis. Default is -1, in which case
        the last axis will be treated as vertical.

    """

    # Initialise the necessary parameters and perform checks
    def __init__(self, data, cov, weights=None, vertaxis=-1):

        self.ndata = np.shape(data)[0]
        self.ndims = np.shape(data)[1]
        self.data = data
        self.cov = cov
        self.weights = np.ones(len(data)) if weights is None else weights
        self.vertaxis = vertaxis

        # Some variables to store the two sets of fitting coordinates and scatter parameters
        self.coords = np.zeros(self.ndims)
        self.normal = np.zeros(self.ndims)
        self.vert_scat = 0.0
        self.norm_scat = 0.0

    # Code to compute normal vectors from cartesian coordinates
    def compute_normal(self, coords=None, vert_scat=None):

        """Converts from data coordinates to the normal vector.

        Keyword Args
        ------------
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

        """Converts from the normal vector to the data coordinates.

        Keyword Args
        ------------
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
    def optimize(self, begin, tol=1.0e-6, verbose=False):

        """Find the best-fitting line/plane/hyperplane.

        Fits the N x D dimensional self.data using scipy.optimise's basinhopping + Nelder-Mead algorithm. Pretty robust.

        Args
        ----
            begin : ndarray
                N + 1 dimensional array of starting values (N parameters of the plane, plus the intrinsic scatter).
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

        Raises
        ------
            ValueError: If the number of values in 'begin' is not equal to N + 1.

        Note
        ----
        If you want to access the best-fitting parameters in the normal coordinates and the scatter normal to the plane,
        these are stored in the self.normal and self.norm_scat class attributes respectively following a call to optimize.

        """

        if len(begin) != self.ndims + 1:
            raise ValueError("length of begin not equal to N dimensions + 1")

        begin = np.array(begin)
        self.normal, self.norm_scat = self.compute_normal(coords=begin[:-1], vert_scat=begin[-1])
        result = basinhopping(
            lambda *args: -self.lnpost(*args),
            np.concatenate([self.normal, [self.norm_scat]]),
            niter=10,
            minimizer_kwargs={"method": "Nelder-Mead", "tol": tol},
        )
        if verbose:
            print(result)
        self.normal = result["x"][:-1]
        self.norm_scat = np.fabs(result["x"][-1])
        self.norm_scat = self.bessel_cochran(self.norm_scat)
        self.coords, self.vert_scat = self.compute_cartesian()

        return self.coords, self.vert_scat

    # A routine run a zeus MCMC on the model given the data
    def zeus(self, begin, max_iter=100000, batchsize=1000, ntau=10.0, tautol=0.05, verbose=False):

        """Run an MCMC on the data using the zeus sampler (Karamanis and Beutler 2020).

        The MCMC runs in batches, checking convergence at the end of each batch until either the chain is well converged
        or the maximum number of iterations has been reached. Convergence is defined as the point when the chain is longer
        than ntau autocorrelation lengths, and the estimate of the autocorrelation length varies less than tautol between batches.
        Burn-in is then removed from the samples, before they are flattened and returned.

        Args
        ----
            begin : ndarray
                N + 1 dimensional array of starting values (N parameters of the plane, plus the intrinsic scatter).
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
        mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1, :], norm_scat=samples[-1, :])).T
        mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return mcmc_samples, mcmc_lnlike

    # A routine run a zeus MCMC on the model given the data
    def emcee(self, begin, max_iter=100000, batchsize=1000, ntau=50.0, tautol=0.05, verbose=False):

        """Run an MCMC on the data using the emcee sampler (Foreman-Mackay et. al., 2013).

        The MCMC runs in batches, checking convergence at the end of each batch until either the chain is well converged
        or the maximum number of iterations has been reached. Convergence is defined as the point when the chain is longer
        than ntau autocorrelation lengths, and the estimate of the autocorrelation length varies less than tautol between batches.
        Burn-in is then removed from the samples, before they are flattened and returned.

        Args
        ----
            begin : ndarray
                N + 1 dimensional array of starting values (N parameters of the plane, plus the intrinsic scatter).
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
        mcmc_samples = np.vstack(self.compute_cartesian(normal=samples[:-1, :], norm_scat=samples[-1, :])).T
        mcmc_lnlike = sampler.get_log_prob(discard=burnin, flat=True)

        return mcmc_samples, mcmc_lnlike


if __name__ == "__main__":

    # Run some tests, copying the examples in Robotham and Obreschkow 2015
    from src.hyperfit.hyperfit import HyperFit
    from src.hyperfit.data import GAMAsmVsize, Hogg, TFR, FP6dFGS, MJB

    data = GAMAsmVsize()
    hf = HyperFit(data.xs, data.cov, weights=data.weights).zeus([1.0, 0.0, 0.0], verbose=False)
    print(np.mean(hf[0], axis=0), np.std(hf[0], axis=0))
    hf = HyperFit(data.xs, data.cov, weights=data.weights).emcee([1.0, 0.0, 0.0], verbose=False)
    print(np.mean(hf[0], axis=0), np.std(hf[0], axis=0))
