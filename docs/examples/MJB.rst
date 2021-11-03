#################################################################
Mass-Spin-Morphology relation from Obreschkow and Glazebrook 2014
#################################################################

Run the fit

.. code-block:: python

    from hyperfit.linfit import LinFit
    from hyperfit.data import MJB

    # Load the data
    data = MJB()
    hf = LinFit(data.xs, data.cov, weights=data.weights)

    # Run an MCMC
    bounds = ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1.0e-5, 1.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=True)
    print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))

Returns

.. math::

    B/T \sim \mathcal{N}[\mu=(0.34 \pm 0.04)\,\mathrm{log_{10}}\mathcal{M} - (0.36 \pm 0.04)\,\mathrm{log_{10}}j - (0.05 \pm 0.02) , \,\sigma=0.02 \pm 0.01]