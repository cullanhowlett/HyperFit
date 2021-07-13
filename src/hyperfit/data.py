from abc import ABC

import os
import inspect
import numpy as np
import pandas as pd


class FitData(ABC):
    """Abstract base class for the test data included with HyperFit

    Meant to only be accessed through the various listed data subclasses. The attributes below are inherited
    by these subclasses

    Attributes
    ----------
    xs: ndarray
        The N x D dimensional data vector
    cov: ndarray
        The N x N x D dimensional set of covariance matrices.
    weights: ndarray, optional
        D dimensional array of weights for each data. Default is None, in which can unit weights are assumed
        for each data point.

    """

    def __init__(self):

        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/data/")

        self.xs = None
        self.cov = None
        self.weights = None


class GAMAsmVsize(FitData):
    """GAMA mass size relation data from Lange et. al., 2015

    Contains 2 x 1854 data of log(mass) in solar masses and log(effective_radius) in kpc
    along with a diagonal covariance matrix (uncorrelated measurement pairs) and weights.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/GAMAsmVsize.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data["logmstar"].to_numpy(), data["logrekpc"].to_numpy()])
        self.cov = np.array(
            [
                [data["logmstar_err"].to_numpy() ** 2, np.zeros(len(data))],
                [np.zeros(len(data)), data["logrekpc_err"].to_numpy() ** 2],
            ]
        )
        self.weights = data["weights"].to_numpy()


class ExampleData(FitData):
    """The random data points used for the tutorial

    Contains 2 x 100 data of dummy x and y values along with a full covariance matrix (correlated measurement pairs)
    and random weights.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/Example.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data[" x"].to_numpy(), data["y"].to_numpy()])
        err = np.array([data["x_err"].to_numpy(), data["y_err"].to_numpy()])
        corr = np.array(
            [[np.ones(len(data)), data["corxy"].to_numpy()], [data["corxy"].to_numpy(), np.ones(len(data))]]
        )
        self.cov = np.einsum("jd,ijd,id->ijd", err, corr, err)
        self.weights = data["weight"].to_numpy()


class TFR(FitData):
    """Tully-Fisher data from Obreschkow and Meyer 2013

    Contains 2 x 55 data of log(maximum_rotation_width) in km/s and absolute K-band magnitude
    along with a diagonal (uncorrelated measurement pairs) covariance matrix and weights.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/TFR.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data["logv"].to_numpy(), data["M_K"].to_numpy()])
        self.cov = np.array(
            [
                [data["logv_err"].to_numpy() ** 2, np.zeros(len(data))],
                [np.zeros(len(data)), data["M_K_err"].to_numpy() ** 2],
            ]
        )
        self.weights = data["weights"].to_numpy()


class FP6dFGS(FitData):
    """6dFGS J-Band Fundamental Plane data from Campbell et. al., 2014 (originally fit in Magoulas et. al., 2012)

    Contains 3 x 8803 data of log(effective_surface_brightness) in L_sol/pc^-2, log(velocity_dispersion) in km/s
    and log(effective_radius) in kpc, along with a full (correlated measurement triplets) covariance matrix and weights.
    Differs slightly from the original data provided with the Robotham and Obreschkow HyperFit package as it
    now includes the cross-correlation coefficient of -0.95 between log(effective_surface_brightness) and
    log(effective_radius) reported in Magoulas et. al., 2012.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/FP6dFGS.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data["logIe_J"].to_numpy(), data["logsigma"].to_numpy(), data["logRe_J"].to_numpy()])
        err = np.array(
            [data["logIe_J_err"].to_numpy(), data["logsigma_err"].to_numpy(), data["logRe_J_err"].to_numpy()]
        )
        corr = np.array(
            [
                [np.ones(len(data)), np.zeros(len(data)), np.zeros(len(data)) - 0.95],
                [np.zeros(len(data)), np.ones(len(data)), np.zeros(len(data))],
                [np.zeros(len(data)) - 0.95, np.zeros(len(data)), np.ones(len(data))],
            ]
        )
        self.cov = np.einsum("jd,ijd,id->ijd", err, corr, err)
        self.weights = data["weights"].to_numpy()

        print(np.shape(self.xs), np.shape(self.cov), np.shape(self.weights))


class MJB(FitData):
    """Mass-Spin-Morphology data from Obreschkow and Glazebrook 2014

    Contains 3 x 16 data of log(disk_baryonic_mass) in 10^10 M_sol, log(angular_momentum) in 10^-3 kpc km/s
    and bulge-to-total ratio along with a full (correlated measurement triplets) covariance matrix and unit weights.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/MJB.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data["logM"].to_numpy(), data["logj"].to_numpy(), data["B/T"].to_numpy()])
        err = np.array([data["logM_err"].to_numpy(), data["logj_err"].to_numpy(), data["B/T_err"].to_numpy()])
        corr = np.array(
            [
                [np.ones(len(data)), data["corMJ"].to_numpy(), np.zeros(len(data))],
                [data["corMJ"].to_numpy(), np.ones(len(data)), np.zeros(len(data))],
                [np.zeros(len(data)), np.zeros(len(data)), np.ones(len(data))],
            ]
        )
        self.cov = np.einsum("jd,ijd->ijd", err, np.einsum("id,ijd->ijd", err, corr))
        self.weights = np.ones(len(data))
