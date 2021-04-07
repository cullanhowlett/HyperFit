HyperFit
========

HyperFit is a simple python package designed to do one thing: fit
N dimensional data with (possibly covariant) errors with an N-1 dimensional plane, and fit it fast.

Put simply, if you have data with x and y errors and you want the correct method for fitting
a straight line, use this package! But it also works for higher dimensions.

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
    zeus
    emcee

HyperFit can be installed via::

    pip install hyperfit

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
