HyperFit
========

HyperFit is a simple python package designed to do one thing:
fit N dimensional data with (possibly covariant) errors with an N-1 dimensional plane, and fit it fast.

Put simply, if you have data with x and y errors and you want the correct method for fitting
a straight line, use this package! But it also works for higher dimensions such as 3D planes and even hyperplanes.

Here is a short example of it in action, which is covered in more detail in the :ref:`tutorial`

.. code-block:: python

    import numpy as np
    from hyperfit import LinFit

    data = ExampleData()
    hf = LinFit(data.xs, data.cov, weights=data.weights)

    bounds = ((-5.0, 5.0), (-10.0, 10.0), (1.0e-5, 5.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=False)
    sigmas = hf.get_sigmas()

which with some additional plotting code, would allow you to produce a figure like below. Some real astrophysical samples
are given in :ref:`examples-index`.

.. image:: Example.png

Contents
--------

.. toctree::
   :maxdepth: 3

   tutorial
   examples/index
   API

Installation
------------

HyperFit requires the following dependencies::

    numpy
    scipy
    pandas
    zeus-mcmc
    emcee

HyperFit can be installed via::

    pip install hyperfit

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
