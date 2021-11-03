###############################################################
6dFGS J-Band Fundamental Plane data from Campbell et. al., 2014
###############################################################

Run the fit

.. code-block:: python

    from hyperfit.linfit import LinFit
    from hyperfit.data import FP6dFGS

    # Load the data
    data = FP6dFGS()
    hf = LinFit(data.xs, data.cov, weights=data.weights)

    # Run an MCMC
    bounds = ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1.0e-5, 1.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=True)
    print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))

Returns

.. math::

    \mathrm{log_{10}}R_{e_{J}} \sim \mathcal{N}[\mu=(0.856 \pm 0.005)\,\mathrm{log_{10}}I_{e_{J}} + (1.507 \pm 0.012)\,\mathrm{log_{10}}\sigma_{v} - (0.41 \pm 0.03) , \\ \,\sigma=0.088 \pm 0.001]