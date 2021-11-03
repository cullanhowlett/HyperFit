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

    def plot(self, linfit=None):
        """Produces a plot of the data where implemented, colour-coded by weight. If a linfit class instance is specified
        it will also plot the best-fit and instead colour code the points by sigma offset

        Args
        ----
            linfit : object, optional
                LinFit class instance from which the best-fit to the data can be accessed, and the sigma offsets
                computed

        Raises
        ------
            NotImplementedError: If called from a subclass with 3-D data (MJB or FP6dFGS)

        """

        return

    def _plot_instance(self, labels, limits, linfit=None):

        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
        from matplotlib.patches import Ellipse

        # Generate ellipses
        ells = [
            Ellipse(
                xy=[self.xs[0, i], self.xs[1, i]],
                width=2.0 * np.sqrt(self.cov[0, 0, i]),
                height=2.0 * np.sqrt(self.cov[1, 1, i]),
                angle=np.rad2deg(np.arcsin(self.cov[0, 1, i] / np.sqrt(self.cov[0, 0, i] * self.cov[1, 1, i]))),
            )
            for i in range(len(self.xs[0]))
        ]

        if linfit is not None:
            xvals = np.linspace(limits[0][0], limits[0][1], 1000)
            yvals = linfit.coords[0] * xvals + linfit.coords[1]
            sigmas = linfit.get_sigmas()

        # Make the plot
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 1.03, 0.83])
        for i, e in enumerate(ells):
            ax.add_artist(e)
            if linfit is not None:
                e.set_color(cm.viridis(sigmas[i] / np.amax(sigmas)))
            else:
                e.set_color(cm.Blues(self.weights[i]))
            e.set_edgecolor("k")
            e.set_alpha(0.9)
        if linfit is not None:
            ax.plot(xvals, yvals, c="k", marker="None", ls="-", lw=1.3, alpha=0.9)
            ax.plot(xvals, yvals - linfit.vert_scat, c="k", marker="None", ls="--", lw=1.3, alpha=0.9)
            ax.plot(xvals, yvals + linfit.vert_scat, c="k", marker="None", ls="--", lw=1.3, alpha=0.9)
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)
        ax.set_xlim(limits[0][0], limits[0][1])
        ax.set_ylim(limits[1][0], limits[1][1])
        ax.tick_params(width=1.3)
        ax.tick_params("both", length=10, which="major")
        ax.tick_params("both", length=5, which="minor")
        for axis in ["top", "left", "bottom", "right"]:
            ax.spines[axis].set_linewidth(1.3)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(12)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(12)

        # Add the colourbar
        if linfit is not None or not np.all(self.weights == self.weights[0]):
            vmax = np.amax(sigmas) if linfit is not None else np.amax(self.weights)
            label = r"$\sigma$" if linfit is not None else r"$\mathrm{weight}$"
            cmap = cm.viridis if linfit is not None else cm.Blues
            cb = fig.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=vmax), cmap=cmap),
                ax=ax,
                shrink=0.55,
                aspect=10,
                anchor=(-7.1, 0.95),
            )
            cb.set_label(label=label, fontsize=14)
            cb.outline.set_linewidth(1.2)
            cb.ax.tick_params(width=1.3, labelsize=11)
            cb.ax.tick_params("both", length=10, which="major")
            cb.ax.tick_params("both", length=5, which="minor")

        plt.show()


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

    def plot(self, linfit=None):
        """Produces a plot of the GAMA size-mass relation. See plot function in FitData base class for more details."""

        labels = [
            r"$\mathrm{log_{10}}(\mathrm{Stellar\,Mass}/M_{\odot})})$",
            r"$\mathrm{log_{10}}(R_{e}/\mathrm{kpc})$",
        ]
        limits = [[8.3, 11.7], [-0.5, 1.3]]
        self._plot_instance(labels, limits, linfit=linfit)


class Hogg(FitData):
    """Data from Hogg et. al., 2010

    Contains 2 x 20 data of dummy x and y values along with a full covariance matrix
    (correlated measurement pairs) and uniform weights.

    This is a subclass that extends FitData but with no additional methods, it
    just inherits the attributes xs, cov and weights detailed in the FitData description.

    """

    def __init__(self):
        super().__init__()

        data = pd.read_csv(self.data_location + f"/Hogg.txt", delim_whitespace=True, escapechar="#")
        self.xs = np.array([data["x"].to_numpy(), data["y"].to_numpy()])
        err = np.array([data["x_err"].to_numpy(), data["y_err"].to_numpy()])
        corr = np.array(
            [[np.ones(len(data)), data["corxy"].to_numpy()], [data["corxy"].to_numpy(), np.ones(len(data))]]
        )
        self.cov = np.einsum("jd,ijd,id->ijd", err, corr, err)
        self.weights = np.ones(len(data))


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

    def plot(self, linfit=None):
        """Produces a plot of the Example data from the tutorial. See plot function in FitData base class for more details."""

        labels = [r"$x$", r"$y$"]
        limits = [[-0.3, 1.3], [0.5, 3.7]]
        self._plot_instance(labels, limits, linfit=linfit)


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

    def plot(self, linfit=None):
        """Produces a plot of the Tully-Fisher data. See plot function in FitData base class for more details."""

        labels = [
            r"$\mathrm{log_{10}}(\mathrm{Velocity}/\mathrm{km\,s^{-1}})})$",
            r"$\mathrm{Absolute\,K}\,(\mathrm{mag})$",
        ]
        limits = [[1.75, 2.5], [-19.0, -26.0]]
        self._plot_instance(labels, limits, linfit=linfit)


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
