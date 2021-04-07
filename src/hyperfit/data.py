from abc import ABC

import os
import inspect
import numpy as np
import pandas as pd


class FitData(ABC):
    """ Abstract base class for the test data included with HyperFit """

    def __init__(self):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/data/")

    def assign_data(self, xs, cov, weights):
        self.xs = xs
        self.cov = cov
        self.weights = weights


class GAMAsmVsize(FitData):
    """ GAMA mass size relation data  """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/GAMAsmVsize.txt", delim_whitespace=True, escapechar="#")
        xs = np.array([data["logmstar"].to_numpy(), data["logrekpc"].to_numpy()]).T
        cov = np.array(
            [
                [data["logmstar_err"].to_numpy() ** 2, np.zeros(len(data))],
                [np.zeros(len(data)), data["logrekpc_err"].to_numpy() ** 2],
            ]
        ).T
        self.assign_data(xs, cov, data["weights"].to_numpy())


class Hogg(FitData):
    """ Data from Hogg et. al., 2010, minus row 3  """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/hogg.txt", delim_whitespace=True, escapechar="#").drop(2)
        xs = np.array([data["x"].to_numpy(), data["y"].to_numpy()]).T
        err = np.array([data["x_err"].to_numpy(), data["y_err"].to_numpy()])
        corr = np.array(
            [[np.ones(len(data)), data["corxy"].to_numpy()], [data["corxy"].to_numpy(), np.ones(len(data))]]
        )
        cov = np.einsum("jd,dij->dij", err, np.einsum("id,ijd->dij", err, corr))
        print(xs, cov)
        self.assign_data(xs, cov, np.ones(len(data)))


class TFR(FitData):
    """ Tully-Fisher data from Obreschkow and Meyer 2013  """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/TFR.txt", delim_whitespace=True, escapechar="#")
        xs = np.array([data["logv"].to_numpy(), data["M_K"].to_numpy()]).T
        cov = np.array(
            [
                [data["logv_err"].to_numpy() ** 2, np.zeros(len(data))],
                [np.zeros(len(data)), data["M_K_err"].to_numpy() ** 2],
            ]
        ).T
        self.assign_data(xs, cov, data["weights"].to_numpy())


class FP6dFGS(FitData):
    """ 6dFGS Fundamental Plane data from Campbell 2014 (originally fit in Magoulas 2012)  """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/FP6dFGS.txt", delim_whitespace=True, escapechar="#")
        xs = np.array([data["logIe_J"].to_numpy(), data["logsigma"].to_numpy(), data["logRe_J"].to_numpy()]).T
        cov = np.array(
            [
                [data["logIe_J_err"].to_numpy() ** 2, np.zeros(len(data)), np.zeros(len(data))],
                [np.zeros(len(data)), data["logsigma_err"].to_numpy() ** 2, np.zeros(len(data))],
                [np.zeros(len(data)), np.zeros(len(data)), data["logRe_J_err"].to_numpy() ** 2],
            ]
        ).T
        self.assign_data(xs, cov, data["weights"].to_numpy())


class MJB(FitData):
    """ Mass-Spin-Morphology data from Obreschkow and Glazebrook 2014 """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/MJB.txt", delim_whitespace=True, escapechar="#")
        xs = np.array([data["logM"].to_numpy(), data["logj"].to_numpy(), data["B/T"].to_numpy()]).T
        err = np.array([data["logM_err"].to_numpy(), data["logj_err"].to_numpy(), data["B/T_err"].to_numpy()])
        corr = np.array(
            [
                [np.ones(len(data)), data["corMJ"].to_numpy(), np.zeros(len(data))],
                [data["corMJ"].to_numpy(), np.ones(len(data)), np.zeros(len(data))],
                [np.zeros(len(data)), np.zeros(len(data)), np.ones(len(data))],
            ]
        )
        cov = np.einsum("jd,dij->dij", err, np.einsum("id,ijd->dij", err, corr))
        self.assign_data(xs, cov, np.ones(len(data)))
