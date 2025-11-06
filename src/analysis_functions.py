import invisible_cities.core.fit_functions as fit
import invisible_cities.core.core_functions as coref

import histograms as hst
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import warnings
from typing import Callable, Union , Optional 
from typing import Tuple, List
from dataclasses import dataclass, field 
from scipy.optimize import curve_fit

from stats import mean_and_std, in_range, poisson_sigma
from plot_functions import color_sequence
from stats import compute_resolution_with_error 
    



@dataclass
class FitFunction:
    fn     : Callable 
    values   : np.array 
    errors  : np.array
    chi2    : float
    pvalue  : float
    cov     : np.array


@dataclass
class FitResult:
    par  : np.array
    err  : np.array
    chi2 : float
    valid : bool


@dataclass
class ProfilePar:
    x  : np.array
    y  : np.array
    xu : np.array
    yu : np.array


@dataclass
class FitPar(ProfilePar):
    f     : FitFunction


@dataclass
class HistoPar:
    var    : np.array
    nbins  : int
    range  : Tuple[float, float]


@dataclass
class FitCollection:
    fp   : FitPar
    hp   : HistoPar
    fr   : FitResult

@dataclass
class GaussPar:
    mu    : float
    std   : float
    amp   : float


@dataclass
class MapPar:
    hratio: np.ndarray
    hcounts: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    zedges: np.ndarray
    hmap: Optional[np.ndarray] = field(default=None)
    norm_val: Optional[float] = field(default=None)


### profiles

def histogram_y_profile(
    h2: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    errMean: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Y-profile statistics from a 2D histogram.

    For each X-bin (row of h2), compute:
    - y_mode: Y bin center with maximum weight (mode)
    - y_mean: weighted mean of Y values
    - y_std: weighted standard deviation of Y values

    Parameters:
    - h2: 2D NumPy array of histogram counts, shape (nx, ny)
    - xedges: 1D array of X bin edges (length nx+1)
    - yedges: 1D array of Y bin edges (length ny+1)

    Returns:
    - xp: 1D array of X bin centers (length nx)
    - y_mode: Y bin center of max weight per X-bin
    - y_mean: weighted mean of Y values per X-bin
    - y_std: weighted std deviation of Y values per X-bin
    """
    
    nx, ny = h2.shape
    xp = 0.5 * (xedges[:-1] + xedges[1:])
    yp = 0.5 * (yedges[:-1] + yedges[1:])

    y_mode, y_mean, y_std = [], [], []

    for i in range(nx):
        weights = h2[i]
        total = np.sum(weights)
        if total > 0:
            max_idx = np.argmax(weights)
            mode = yp[max_idx]
            mean = np.average(yp, weights=weights)
            var = np.average((yp - mean) ** 2, weights=weights)
            std = np.sqrt(var)
            if errMean:
                std = std/np.sqrt(total)
        else:
            mode, mean, std = np.nan, np.nan, np.nan

        y_mode.append(mode)
        y_mean.append(mean)
        y_std.append(std)

    return np.array(xp), np.array(y_mode), np.array(y_mean), np.array(y_std)


### Selection functions

def road_dt_zrms2(
    dfs1: pd.DataFrame,
    nbins: int,
    xrange: Union[List[float], np.array],
    yrange: Union[List[float], np.array],
    figsize: Tuple[int, int] = (10, 4),
    nsigma: float = 2.0,
    errMean: bool = False,
    xlabel: str = "DT",
    ylabel: str = "Zrms²",
    title: str = "Road Band"
) -> Tuple[FitFunction, FitFunction, FitFunction, pd.DataFrame]:
    """
    Build a band ("road") around the most probable Zrms² value in each DT bin and filter points within it.

    Parameters
    ----------
    dfs1 : pd.DataFrame
        DataFrame with 'DT' and 'Zrms' columns.
    nbins : int
        Number of bins for 2D histogram.
    xrange, yrange : list of float
        Histogram ranges for X (DT) and Y (Zrms²).
    figsize : tuple, optional
        Size of the matplotlib figure.
    nsigma : float, optional
        Multiplier of standard deviation to define the road band.
    errMean : bool, optional
        Use standard error instead of standard deviation per bin.
    xlabel, ylabel, title : str
        Axis and plot labels.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with points inside the road band.
    """
    h2, xedges, yedges = np.histogram2d(
        dfs1.DT, dfs1.Zrms**2, bins=(nbins, nbins), range=[xrange, yrange]
    )

    fig, axs = plt.subplots(figsize=figsize)
    hst.plot_h2d(fig, axs, h2, xedges, yedges, xlabel, ylabel, title)

    xp, yp, _, ystd = histogram_y_profile(h2, xedges, yedges, errMean)

    yl = yp - nsigma * ystd
    yr = yp + nsigma * ystd

    ff_c = fit.fit(fit.polynom, xp, yp, [1., 1.])
    ff_l = fit.fit(fit.polynom, xp, yl, [1., 1.])
    ff_u = fit.fit(fit.polynom, xp, yr, [1., 1.])

    axs.errorbar(xp, yp, yerr=ystd, fmt='o', ms=3, elinewidth=2, capsize=3, color='black')
    axs.plot(xp, ff_c.fn(xp), 'red', lw=2, label='Central Fit')
    axs.plot(xp, ff_l.fn(xp), 'blue', lw=2, label=f'-{nsigma}σ Fit')
    axs.plot(xp, ff_u.fn(xp), 'green', lw=2, label=f'+{nsigma}σ Fit')

    axs.legend()
    fig.tight_layout()
    plt.show()

    x = dfs1.DT.to_numpy()
    y = dfs1.Zrms.to_numpy() ** 2

    lower_bound = ff_l.fn(x)
    upper_bound = ff_u.fn(x)
    mask = (y >= lower_bound) & (y <= upper_bound)

    

    return (FitFunction(ff_c.fn, ff_c.values, ff_c.errors, ff_c.chi2, ff_c.pvalue, ff_c.cov), 
            FitFunction(ff_l.fn, ff_l.values, ff_l.errors, ff_l.chi2, ff_l.pvalue, ff_l.cov), 
            FitFunction(ff_u.fn, ff_u.values, ff_u.errors, ff_u.chi2, ff_u.pvalue, ff_u.cov), 
            dfs1[mask].copy())


def road_r_s2e(
    dfs1: pd.DataFrame,
    nbins: int,
    xrange: Union[List[float], np.array],
    yrange: Union[List[float], np.array],
    figsize: Tuple[int, int] = (10, 4),
    nsigma: float = 2.0,
    errMean: bool = False,
    xlabel: str = "DT",
    ylabel: str = "Zrms²",
    title: str = "Road Band"
) -> Tuple[FitFunction, FitFunction, FitFunction, pd.DataFrame]:
    """
    Build a band ("road") around the most probable S2e value in each R bin and filter points within it.

    Parameters
    ----------
    dfs1 : pd.DataFrame
        DataFrame with 'R' and 'S2e' columns.
    nbins : int
        Number of bins for 2D histogram.
    xrange : list of float
        Histogram range for X (R).
    yrange : list of float
        Histogram range for Y (S2e).
    figsize : tuple, optional
        Size of the matplotlib figure.
    nsigma : float, optional
        Number of standard deviations for defining the band.
    errMean : bool, optional
        If True, use standard error instead of std deviation.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    title : str, optional
        Title for the plot.

    Returns
    -------
    Tuple[object, object, object, pd.DataFrame]
        Tuple containing:
        - Central polynomial fit object
        - Lower fit object (-nsigma)
        - Upper fit object (+nsigma)
        - Filtered DataFrame within the band
    """
    h2, xedges, yedges = np.histogram2d(
        dfs1.R, dfs1.S2e, bins=(nbins, nbins), range=[xrange, yrange]
    )

    fig, axs = plt.subplots(figsize=figsize)
    hst.plot_h2d(fig, axs, h2, xedges, yedges, xlabel, ylabel, title)

    xp, yp, _, ystd = histogram_y_profile(h2, xedges, yedges, errMean)

    mask_valid = np.isfinite(yp) & np.isfinite(ystd)
    xp = xp[mask_valid]
    yp = yp[mask_valid]
    ystd = ystd[mask_valid]

    if len(xp) < 4:
        raise ValueError("Not enough valid points to fit a 3rd-degree polynomial.")

    yl = yp - nsigma * ystd
    yr = yp + nsigma * ystd

    ff_c = fit.fit(fit.polynom, xp, yp, [1., 1., 1., 1.])
    ff_l = fit.fit(fit.polynom, xp, yl, [1., 1., 1., 1.])
    ff_u = fit.fit(fit.polynom, xp, yr, [1., 1., 1., 1.])

    axs.errorbar(xp, yp, yerr=ystd, fmt='o', ms=3, elinewidth=2, capsize=3, color='black')
    axs.plot(xp, ff_c.fn(xp), 'red', lw=2, label='Central Fit')
    axs.plot(xp, ff_l.fn(xp), 'blue', lw=2, label=f'-{nsigma}σ Fit')
    axs.plot(xp, ff_u.fn(xp), 'green', lw=2, label=f'+{nsigma}σ Fit')

    axs.legend()
    fig.tight_layout()
    plt.show()

    x = dfs1.R.to_numpy()
    y = dfs1.S2e.to_numpy()

    lower_bound = ff_l.fn(x)
    upper_bound = ff_u.fn(x)
    mask = (y >= lower_bound) & (y <= upper_bound)

    return (FitFunction(ff_c.fn, ff_c.values, ff_c.errors, ff_c.chi2, ff_c.pvalue, ff_c.cov), 
            FitFunction(ff_l.fn, ff_l.values, ff_l.errors, ff_l.chi2, ff_l.pvalue, ff_l.cov), 
            FitFunction(ff_u.fn, ff_u.values, ff_u.errors, ff_u.chi2, ff_u.pvalue, ff_u.cov), 
            dfs1[mask].copy())

#maos


def map3d_norm(map3d, xedges, yedges, radius=50.0):
    """
    Compute the mean value in a circular region of radius `radius` (in mm)
    centered at (0, 0) in the XY plane for the first Z-slice of the 3D map.

    Parameters:
        map3d (ndarray): 3D array of shape (Nx, Ny, Nz)
        xedges (ndarray): 1D array of bin edges along X
        yedges (ndarray): 1D array of bin edges along Y
        radius (float): Radius around (0, 0) in mm

    Returns:
        float: Mean value within the circular region
    """
    slice_xy = map3d[:, :, 0]

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    R = np.sqrt(X**2 + Y**2)
    mask = R <= radius

    values = slice_xy[mask]
    values = values[~np.isnan(values)]

    return np.mean(values) if values.size > 0 else np.nan



def compute_map3D(df: pd.DataFrame, bins_xy: int=100, 
                  z_bins=Optional[List[float]], 
                  xmin: float =-480.0, xmax: float =480.0)->MapPar:
    """
    Compute a 3D map of mean S2e over X, Y, Z bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'X', 'Y', 'DT', 'S2e'.
    bins_xy : int, optional
        Number of bins for X and Y axes (uniform).
    z_bins : array-like, optional
        Custom bin edges for Z-axis (DT).
    xmin : float, optional
        Minimum for X and Y axes.
    xmax : float, optional
        Maximum for X and Y axes.

    Returns
    -------
    a MapPar() with
    hist_ratio : np.ndarray
        3D histogram of mean S2e values.
    hist_counts : np.ndarray
        3D histogram of counts values in bins.
    edges_x, edges_y, edges_z : np.ndarray
        Bin edges for X, Y, Z.
    hmap : np.ndarray
        3D histogram with the correction values normalized with map3d_norm() ->Return the average in the fist z slice in a radius of 50 mm.
    """
    if z_bins is None:
        zmin = df['DT'].min()
        zmax = df['DT'].max()
        z_bins = np.linspace(zmin, zmax, 11)  # default 10 bins

    x_bins = np.linspace(xmin, xmax, bins_xy + 1)
    y_bins = np.linspace(xmin, xmax, bins_xy + 1)

    hist_S2e, (edges_x, edges_y, edges_z) = np.histogramdd(
        (df['X'], df['Y'], df['DT']),
        bins=(x_bins, y_bins, z_bins),
        weights=df['S2e']
    )

    hist_counts, _ = np.histogramdd(
        (df['X'], df['Y'], df['DT']),
        bins=(x_bins, y_bins, z_bins)
    )

    hist_ratio = np.divide(hist_S2e, hist_counts, where=hist_counts != 0)

    norm_factor = map3d_norm(hist_ratio,edges_x,edges_y)

    hmap = np.divide(hist_ratio, norm_factor, where=hist_counts > 2)
    
    return MapPar(hratio=hist_ratio, hcounts=hist_counts, 
                  xedges=edges_x, yedges=edges_y, zedges=edges_z, hmap=hmap, norm_val=norm_factor )


def correct_S2e(
    df: pd.DataFrame,
    krmap: MapPar,
    rmax: float = 480.0,
    zmax: float = 1350.0
) -> pd.DataFrame:
    """
    Apply position-based S2e correction using a 3D histogram map, safely guarding against divide-by-zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'X', 'Y', 'DT', 'R', 'S2e'.
    krmap : MapPar
        Correction map object containing hmap and bin edges.
    rmax : float, optional
        Max radial cut.
    zmax : float, optional
        Max drift time cut.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Epes' and 'corrections' columns.
    """
    dfc = df[(df['DT'] < zmax) & (df['R'] < rmax)].copy()

    x_vals = dfc['X'].to_numpy()
    y_vals = dfc['Y'].to_numpy()
    z_vals = dfc['DT'].to_numpy()
    s2e_vals = dfc['S2e'].to_numpy()

    x_bins = np.digitize(x_vals, krmap.xedges) - 1
    y_bins = np.digitize(y_vals, krmap.yedges) - 1
    z_bins = np.digitize(z_vals, krmap.zedges) - 1

    hmap = krmap.hmap
    if hmap is None:
        raise ValueError("krmap.hmap is None. You must set krmap.hmap before calling correct_S2e().")

    valid_mask = (
        (x_bins >= 0) & (x_bins < hmap.shape[0]) &
        (y_bins >= 0) & (y_bins < hmap.shape[1]) &
        (z_bins >= 0) & (z_bins < hmap.shape[2])
    )

    corrections = np.full_like(s2e_vals, np.nan)
    corrections[valid_mask] = hmap[x_bins[valid_mask], y_bins[valid_mask], z_bins[valid_mask]]

    # Avoid divide-by-zero and negative corrections
    safe_mask = corrections > 1e-6
    norm_s2e = np.full_like(s2e_vals, np.nan)
    norm_s2e[safe_mask] = s2e_vals[safe_mask] / corrections[safe_mask]

    dfc['Epes'] = norm_s2e
    dfc['corrections'] = corrections

    dfc = dfc.dropna(subset=['Epes'])

    return dfc


### fits


def fit_lifetime(
    dfs1: pd.DataFrame,
    nbins: int = 15,
    dtrange: List[float] = [200.0, 1200.0],
    s2range: List[float] = [4000, 4800],
    stdmode: str = "mode",
    nsigma: float = 5.0,
    figsize: Tuple[int, int] = (10, 4)
) -> Tuple[float, float]:
    """
    Fit an exponential decay model to the S2 area (S2e) as a function of drift time (DT),
    using a two-pass filter on histogram bin counts.

    Parameters
    ----------
    dfs1 : pd.DataFrame
        DataFrame with 'DT' and 'S2e' columns.
    nbins : int
        Histogram bin count.
    dtrange, s2range : list of float
        Histogram edges for DT and S2e.
    stdmode : str
        Method to estimate sigma (e.g., 'mode' or 'std').
    nsigma : float
        Scale factor for filtering range in first profile.
    figsize : tuple
        Size of the matplotlib figure.

    Returns
    -------
    Tuple[float, float]
        Fitted (const, lambda)
    """
    def expo_seed(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        const = y_sorted[0]
        delta_x = x_sorted[-1] - x_sorted[0]
        decay = delta_x / (np.log((y_sorted[0] + eps) / (y_sorted[-1] + eps)))
        return const, decay

    h2, xedges, yedges = np.histogram2d(
        dfs1.DT, dfs1.S2e, bins=(nbins, nbins), range=[dtrange, s2range]
    )

    fig, axs = plt.subplots(figsize=figsize)
    hst.plot_h2d(fig, axs, h2, xedges, yedges, xlabel="DT", ylabel="S2e", title="Lifetime")

    # Step 1: initial profile
    xp, yp, _, ystd = histogram_y_profile(h2, xedges, yedges)

    # Step 2: apply bin-level filtering to h2
    yp_bounds_min = yp - nsigma * ystd
    yp_bounds_max = yp + nsigma * ystd

    # Calculate bin centers
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    # Mask bins outside [yp - nsigma*ystd, yp + nsigma*ystd]
    for i, (ymin, ymax) in enumerate(zip(yp_bounds_min, yp_bounds_max)):
        mask = (y_centers < ymin) | (y_centers > ymax)
        h2[i, mask] = 0  # suppress contribution outside filtered band

    # Recalculate profile
    xp, yp, _, ystd = histogram_y_profile(h2, xedges, yedges)

    valid_points = ystd > 0
    x = xp[valid_points]
    y = yp[valid_points]

    yu = np.sqrt(y) if stdmode == "mode" else ystd[valid_points]

    seed = expo_seed(x, y)
    ff = fit.fit(fit.expo, x, y, seed, sigma=yu)

    const, lamda = ff.values

    plt.errorbar(x, y, yerr=yu, fmt='o', ms=3, elinewidth=2, capsize=3, color='black')
    plt.plot(x, ff.fn(x), "r-", lw=2)
    plt.ylim(bottom=s2range[0], top=s2range[1])

    # Annotate fit
    c0 = f"constant = {const:7.2f} ± {ff.errors[0]:7.3f} pes"
    c1 = f"λ = {-lamda / 1e3:7.2f} ± {ff.errors[1] / 1e3:7.3f} ms"

    plt.gca().text(
        0.05, 0.95, f"{c0}\n{c1}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    plt.xlabel('DT')
    plt.ylabel('S2e')
    plt.title('Fit S2e vs DT')
    plt.show()

    return const, lamda


def fit_lifetime2(
    dfs1: pd.DataFrame,
    nbins: int = 15,
    dtrange: List[float] = [200.0, 1200.0],
    s2range: List[float] = [4000, 4800],
    stdmode: str = "mode",
    figsize: Tuple[int, int] = (10, 4)
) -> Tuple[float, float]:
    """
    Fit an exponential decay model to the S2 area (S2e) as a function of drift time (DT).

    The function profiles `DT` vs `S2e` and fits the model:
        S2e(DT) ≈ const * exp(-DT / τ)

    Parameters
    ----------
    dfs1 : pd.DataFrame
        Input dataframe with columns `DT` (drift time) and `S2e` (S2 signal area).
    nbins : int, optional
        Number of bins used for profile histogram.
    dtrange : list of float, optional
        Drift time range [min, max] for histogramming.
    s2range : list of float, optional
        S2 area range [min, max] for histogramming.
    figsize : tuple, optional
        Size of the matplotlib figure.

    Returns
    -------
    Tuple[float, float]
        Fitted parameters (const, lambda), where:
            - const: initial S2e amplitude
            - lambda: exponential decay rate (units: 1/μs)
    """
    def expo_seed(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
        """
        Estimate initial parameters for exponential decay fit.

        Returns
        -------
        Tuple[float, float]
            (initial amplitude, decay constant)
        """
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        const = y_sorted[0]
        delta_x = x_sorted[-1] - x_sorted[0]
        decay = delta_x / (np.log((y_sorted[0] + eps) / (y_sorted[-1] + eps)))
        return const, decay

    h2, xedges, yedges = np.histogram2d(
        dfs1.DT, dfs1.S2e, bins=(nbins, nbins), range=[dtrange, s2range]
    )

    fig, axs = plt.subplots(figsize=figsize)
    hst.plot_h2d(fig, axs, h2, xedges, yedges, xlabel="DT", ylabel="S2e", title="Lifetime")

    xp, yp, _, ystd = histogram_y_profile(h2, xedges, yedges)

    valid_points = ystd > 0
    x = xp[valid_points]
    y = yp[valid_points]

    if stdmode == "mode":
        yu = np.sqrt(y)
    else: 
        yu = ystd[valid_points]
    

    seed = expo_seed(x, y)
    ff = fit.fit(fit.expo, x, y, seed, sigma=yu)

    const, lamda = ff.values

    plt.errorbar(x, y, yerr=yu, fmt='o', ms=3, elinewidth=2, capsize=3, color='black')
    plt.plot(x, ff.fn(x), "r-", lw=2)
    plt.ylim(bottom=s2range[0], top=s2range[1])

    # Annotate fit
    c0 = f"constant = {const:7.2f} ± {ff.errors[0]:7.3f} pes"
    c1 = f"λ = {-lamda / 1e3:7.2f} ± {ff.errors[1] / 1e3:7.3f} ms"

    plt.gca().text(
        0.05, 0.95, f"{c0}\n{c1}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    plt.xlabel('DT')
    plt.ylabel('S2e')
    plt.title('Fit S2e vs DT')
    plt.show()

    return const, lamda

#Fit Lifetime with radius semection
def fit_lifetime_Rsel(df : pd.DataFrame,
                  nbins : int = 100,
                  dtrange: List[float] = [50.0, 1350.0],
                  s2range: List[float] = [6000, 10000],
                  radius_selection : float = 250,
                  slices_profile : int = 16
                  ):

    """
    Analyzes S2e as a function of drift time (DT) to extract the electron lifetime
    via an exponential decay fit.
    
    Parameters:
    df : pd.DataFrame
        Input DataFrame 
    nbins : int, default=100
        Number of bins for both axes in the 2D histogram.
    dtrange : List[float], default=[50.0, 1350.0]
        Drift time range 
    s2range : List[float], default=[6000, 10000]
        S2e range 
    radius_selection : float, default=250
        Maximum radial distance (in mm) to select events for the fit.
    
    Output:
    - Displays a plot showing the 2D histogram of S2e vs DT, Gaussian profile points with error bars,
      and the fitted exponential curve with fit parameters A and τ (electron lifetime) displayed in the legend.
    - Return the lifetime with error
    """
    
    # Define exponential function for the fit
    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    # Compute 2D histogram
    counts, xedges, yedges = np.histogram2d(
        df[df['R'] < radius_selection]["DT"], df[df['R'] < radius_selection]["S2e"],
        bins=(nbins,nbins),
        range=(dtrange, s2range)
    )
    
    # Recompute profiles for plotting
    dt_centers, mean_vals, mean_errs, sigma, sigma_errs = gaussian_profiler_y_slices(counts, xedges, yedges, slices_profile)
    
    # Fit the exponential to the mean values
    popt, pcov = curve_fit(exp_decay, dt_centers, mean_vals, p0=[8000,30000], sigma=mean_errs, absolute_sigma=True)
    A_fit, tau_fit = popt
    
    # Evaluate fit
    t_fit = np.linspace(dt_centers.min(), dt_centers.max(), 300)
    y_fit = exp_decay(t_fit, *popt)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(
        counts.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )
    plt.errorbar(dt_centers, mean_vals, yerr=mean_errs, fmt='.', markersize=5, linewidth=1., color='w', label='Profiles')
    plt.plot(t_fit, y_fit, 'r--', linewidth =1.2, label=f'Fit:A={A_fit:.1f}, τ={tau_fit/1000:.1f} ms')
    
    plt.colorbar(label='Counts')
    plt.xlabel('Drift Time (µs)')
    plt.ylabel('S2e (pes)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    perr = np.sqrt(np.diag(pcov))  # Standard errors
    A_err, tau_err = perr

    return tau_fit, tau_err

## gaussian

def gauss_seed(x, y, sigma_rel=0.05):
    """
    Estimate initial parameters (seed) for a Gaussian fit to 1D histogram data.

    Parameters
    ----------
    x : np.array
        Bin centers.
    y : np.array
        Bin counts (heights) corresponding to x.
    sigma_rel : float, optional
        Relative estimate of sigma as a fraction of peak x-value (default is 0.05).

    Returns
    -------
    tuple
        Seed tuple (amplitude, mean, sigma) suitable for Gaussian fitting.
    """
    i_max = np.argmax(y)       # Index of peak bin
    x_max = x[i_max]           # x at peak
    y_max = y[i_max]           # y at peak
    sigma = sigma_rel * x_max  # Estimate of standard deviation
    dx    = np.diff(x)[0]      # Bin width (assumes uniform bins)
    amp   = y_max * np.sqrt(2 * np.pi) * sigma / dx  # Estimate Gaussian area amplitude
    return amp, x_max, sigma

def gaussfl_seed(x, y, sigma_rel=0.01):
    """
    Estimate initial parameters (seed) for a Gaussian + flat term fit to 1D histogram data.

    Parameters
    ----------
    x : np.array
        Bin centers.
    y : np.array
        Bin counts (heights) corresponding to x.
    sigma_rel : float, optional
        Relative estimate of sigma as a fraction of peak x-value (default is 0.05).

    Returns
    -------
    tuple
        Seed tuple (amplitude, mean, sigma) suitable for Gaussian fitting.
    """
    i_min = np.argmin(y)
    i_max = np.argmax(y)       # Index of peak bin
    x_max = x[i_max]           # x at peak
    y_max = y[i_max]           # y at peak
    sigma = sigma_rel * x_max  # Estimate of standard deviation
    dx    = np.diff(x)[0]      # Bin width (assumes uniform bins)
    amp   = y_max * np.sqrt(2 * np.pi) * sigma / dx  # Estimate Gaussian area amplitude
    C = y[i_min]
    return amp, x_max, sigma, C


def gaussian_parameters(x: np.array, xrange: Tuple[float, float], bin_size: float = 1) -> Tuple[float, float, float]:
    """
    Estimate the parameters of a normalized Gaussian distribution from data.

    The Gaussian is defined as:
        g(x) = A * exp(-(x - μ)^2 / (2 * σ^2))

    where:
        - μ is the mean of the distribution,
        - σ is the standard deviation,
        - A is the amplitude (peak height).

    This function computes:
        - μ and σ from the data `x` restricted to `xrange`,
        - A based on total count normalization: A ≈ N / (sqrt(2π) * σ), 
          where N is the number of events scaled by bin size.

    Parameters
    ----------
    x : np.array
        Input data array.
    xrange : Tuple[float, float]
        Range (min, max) over which to calculate mean and standard deviation.
    bin_size : float, optional
        Histogram bin width for normalization (default is 1).

    Returns
    -------
    Tuple[float, float, float]
        (amplitude, mean, standard deviation)
    """
    mu, std = mean_and_std(x, xrange)
    norm = np.sqrt(2 * np.pi) * std
    amp = len(x) * bin_size / norm
    return amp, mu, std


def fit_energy(e       : np.array,
               nbins   : int,
               xrange  : Tuple[float, float],
               n_sigma : float = 3.0) -> Tuple[FitPar, FitResult]:
    """
    Perform a Gaussian fit on a 1D energy spectrum.

    This function:
      1. Computes a histogram of the energy array `e` with `nbins` within `xrange`.
      2. Estimates initial parameters (amplitude, mean, sigma) for a Gaussian.
      3. Fits a Gaussian function to the histogram data, restricting the fit to a 
         window centered on the estimated peak (± `n_sigma` * σ).
      4. Returns structured fit and histogram objects for further analysis or plotting.

    Parameters
    ----------
    e : np.array
        Input energy values.
    nbins : int
        Number of bins in the histogram.
    xrange : Tuple[float, float]
        Range over which to compute the histogram (min, max).
    n_sigma : float, optional
        Number of standard deviations to use for defining the fit window 
        around the estimated mean (default is 3.0).

    Returns
    -------
    Tuple[FitPar, FitResult]
        - FitPar   : structure holding x, y, errors, and the fitted function
        - FitResult: structure holding fit values, errors, chi², and validity
    """

    # Histogram and bin centers
    y, b = np.histogram(e, bins=nbins, range=xrange)
    x = coref.shift_to_bin_centers(b)
    bin_size = (xrange[1] - xrange[0]) / nbins

    # Estimate Gaussian parameters
    amp, mu, std = gaussian_parameters(e, xrange, bin_size)
    fit_range = mu - n_sigma * std, mu + n_sigma * std

    # Restrict fit range
    mask = in_range(x, *fit_range)
    x, y = x[mask], y[mask]
    yu = poisson_sigma(y)

    # Perform fit
    f = fit.fit(fit.gauss, x, y, (amp, mu, std), sigma=yu)
    c2 = chi2(f, x, y, yu)

    # Package results
    fr = FitResult(
        par=np.array(f.values),
        err=np.array(f.errors),
        chi2=c2,
        valid=True
    )

    fp = FitPar(
        x=x,
        y=y,
        xu=np.diff(x) * 0.5,
        yu=yu,
        f=f.fn
    )

    return fp, fr



def fit_Epes_vs_R(
    df: pd.DataFrame,
    r_edges: Optional[List[float]] = None,
    nbins: int = 50,
    epes_range: Optional[List[float]] = None,
    n_sigma: float = 3.0
) -> List:
    """
    Fit Epes distributions in slices of radial position (R).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'R' and 'Epes'.
    r_edges : list of float, optional
        Bin edges for R slicing. Default: 10 bins from 0 to 480 mm.
    nbins : int
        Histogram bins for Epes in each slice.
    epes_range : list of float, optional
        Epes range for histogramming and fitting.
    n_sigma : float
        Sigma cutoff for energy fit.

    Returns
    -------
    List
        List of FitCollection objects, one per R slice.
    """
    if r_edges is None:
        r_edges = np.linspace(0, 480, 11).tolist()
    if epes_range is None:
        epes_range = [7800, 9000]

    fits = []

    for rmin, rmax in zip(r_edges[:-1], r_edges[1:]):
        df_slice = df[(df.R >= rmin) & (df.R < rmax)]
        if len(df_slice) < 10:
            continue  # skip slices with too few entries

        hp = HistoPar(var=df_slice.Epes, nbins=nbins, range=epes_range)
        fp, fr = fit_energy(hp.var, hp.nbins, hp.range, n_sigma=n_sigma)
        fc = FitCollection(fp, hp, fr)
        fits.append(fc)

    return fits



## aux

def plot_fit_energy(fc : FitCollection, label = None):

    if fc.fr.valid:
        par  = fc.fr.par
        x    = fc.hp.var
        r    = 2.35 * 100 *  par[2] / par[1]
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(par[1])
        sigma    =  r'$\sigma$ = {:7.2f}'.format(par[2])
        rx       =  r'$\sigma/mu$ (FWHM)  = {:7.2f}'.format(r)
        stat     =  f'{entries}\n{mean}\n{sigma}\n{rx}'

        _, _, _   = plt.hist(fc.hp.var,
                             bins = fc.hp.nbins,
                             range=fc.hp.range,
                             histtype='step',
                             edgecolor='black',
                             linewidth=1.5,
                             label=label)

        plt.plot(fc.fp.x, fc.fp.f(fc.fp.x), "r-",  linewidth=1.5, label=stat)
        plt.legend()
        
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def plot_fit_energy_list(
    fc_list: List[FitCollection],
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    suptitle: Optional[str] = "Energy fits in R slices"
) -> None:
    """
    Plot multiple energy fits from a list of FitCollection objects.

    Parameters
    ----------
    fc_list : List[FitCollection]
        List of fit results to plot.
    ncols : int, optional
        Number of subplot columns (default 3).
    figsize : tuple, optional
        Overall figure size.
    suptitle : str, optional
        Super title for the figure.
    """
    n = len(fc_list)
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (fc, ax) in enumerate(zip(fc_list, axes)):
        if fc.fr.valid:
            par = fc.fr.par
            x = fc.hp.var
            r = 2.35 * 100 * par[2] / par[1]
            entries = f'Entries = {len(x)}'
            mean = r'$\mu$ = {:7.2f}'.format(par[1])
            sigma = r'$\sigma$ = {:7.2f}'.format(par[2])
            rx = r'$\sigma/\mu$ (FWHM) = {:7.2f}'.format(r)
            stat = f'{entries}\n{mean}\n{sigma}\n{rx}'

            ax.hist(x, bins=fc.hp.nbins, range=fc.hp.range,
                    histtype='step', edgecolor='black',
                    linewidth=1.5, label=stat)
            ax.plot(fc.fp.x, fc.fp.f(fc.fp.x), 'r-', lw=2)
            ax.legend(fontsize='small')
        else:
            warnings.warn(f'Fit {i} failed. Skipping.', UserWarning)
            ax.text(0.5, 0.5, 'Invalid fit', ha='center', va='center')
        
        ax.set_title(f'R Slice {i}', fontsize=10)
        ax.set_xlabel('Epes')
        ax.set_ylabel('Counts')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_resolution_vs_R(
    fc_list: List[FitCollection],
    r_edges: Optional[List[float]] = None,
    title: str = "Resolution (FWHM) vs R",
    ymin: float = 0.0,
    ymax: float = 7.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot sigma/mu (FWHM in %) vs R from a list of FitCollection objects.

    Parameters
    ----------
    fc_list : List[FitCollection]
        List of fit results.
    r_edges : List[float], optional
        Bin edges for R slices (length = len(fc_list) + 1).
    title : str
        Plot title.
    ymin : float
        Minimum y-axis value.
    ymax : float
        Maximum y-axis value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (r_centers, fwhm_vals, fwhm_errs)
    """
    r_centers = []
    fwhm_vals = []
    fwhm_errs = []

    for i, fc in enumerate(fc_list):
        if not fc.fr.valid:
            continue

        par = fc.fr.par
        err = fc.fr.err

        mu, sigma = par[1], par[2]
        d_mu, d_sigma = err[1], err[2]

        if mu <= 0 or sigma <= 0:
            continue

        fwhm = 2.35 * 100 * sigma / mu
        dfwhm = fwhm * np.sqrt((d_mu / mu) ** 2 + (d_sigma / sigma) ** 2)

        if r_edges and len(r_edges) == len(fc_list) + 1:
            r_center = 0.5 * (r_edges[i] + r_edges[i + 1])
        else:
            r_center = i

        r_centers.append(r_center)
        fwhm_vals.append(fwhm)
        fwhm_errs.append(dfwhm)

    r_centers = np.array(r_centers)
    fwhm_vals = np.array(fwhm_vals)
    fwhm_errs = np.array(fwhm_errs)

    plt.errorbar(r_centers, fwhm_vals, yerr=fwhm_errs, fmt='o', capsize=4, lw=1.5, color='black')
    plt.xlabel("R [mm]")
    plt.ylabel("FWHM (%)")
    plt.title(title)
    plt.grid(True)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()

    return r_centers, fwhm_vals, fwhm_errs



def print_fit_energy(fc : FitCollection):

    par  = fc.fr.par
    err  = fc.fr.err
    try:
        r  = 2.35 * 100 *  par[2] / par[1]
        fe = np.sqrt(41 / 2458) * r
        print(f'  Fit was valid = {fc.fr.valid}')
        print(f' Emu       = {par[1]} +-{err[1]} ')
        print(f' E sigma   = {par[2]} +-{err[2]} ')
        print(f' chi2    = {fc.fr.chi2} ')

        print(f' sigma E/E (FWHM)     (%) ={r}')
        print(f' sigma E/E (FWHM) Qbb (%) ={fe} ')
    except ZeroDivisionError:
        warnings.warn(f' mu  = {par[1]} ', UserWarning)


def plot_3d_histogram_slices(
    H: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    zedges: np.ndarray,
    z_indices: list,
    figsize: tuple = (4, 4),
    cmap: str = "jet"
):
    """
    Plot 2D X-Y histograms from slices of a 3D histogram H along Z-axis.

    Parameters
    ----------
    H : np.ndarray
        3D histogram array of shape (nx, ny, nz).
    xedges, yedges, zedges : np.ndarray
        Bin edges for X, Y, and Z dimensions.
    z_indices : list of int
        Indices along the Z-axis to slice and plot.
    figsize : tuple, optional
        Size of each subplot.
    cmap : str, optional
        Colormap used for plotting.

    Returns
    -------
    None
    """

    n_plots = len(z_indices)
    if n_plots <= 2:
        rows, cols = 1, n_plots
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 8:
        rows, cols = 2, 4
    else:
        cols = 4
        rows = (n_plots + 2) // 4

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, idx in enumerate(z_indices):
        ax = axes[i]
        H2d = H[:, :, idx]
        mesh = ax.pcolormesh(
            xedges,
            yedges,
            H2d.T,
            shading='auto',
            cmap=cmap
        )
        ax.set_title(f"Z bin {idx} ({zedges[idx]:.2f} to {zedges[idx+1]:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(mesh, ax=ax)

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    

    ## chi2
def chi2f(f   : Callable,
          nfp : int,
          x   : np.array,
          y   : np.array,
          yu  : np.array) -> float:
    """
    Compute the reduced chi-squared (χ²/ndof) between model predictions and data.

    Parameters
    ----------
    f : Callable
        A function that takes x-values and returns model predictions.
    nfp : int
        Number of free parameters in the fit (used to compute degrees of freedom).
    x : np.array
        Independent variable data points.
    y : np.array
        Dependent variable (measured values).
    yu : np.array
        Uncertainties (standard deviations) associated with y.

    Returns
    -------
    float
        Reduced chi-squared value (χ² / degrees of freedom), or raw χ² if ndof = 0.

    Notes
    -----
    The function assumes all arrays are of the same length and performs:
        χ² = Σ[(yᵢ - f(xᵢ))² / σᵢ²]
    """
    assert len(x) == len(y) == len(yu), "Input arrays must have the same length."

    fitx = f(x)
    residuals_sq = ((y - fitx) / yu) ** 2
    chi2_ = np.sum(residuals_sq)

    ndof = len(x) - nfp
    if ndof > 0:
        return chi2_ / ndof
    else:
        warnings.warn(f"ndof = 0 in chi2 calculation, returning raw chi2 = {chi2_:.3f}", UserWarning)
        return chi2_
    

def chi2(f : FitFunction,
         x : np.array,
         y : np.array,
         sy: np.array) -> float:
    """
    Compute the reduced chi-squared (χ²/ndof) of a FitFunction object against data.

    Parameters
    ----------
    f : FitFunction
        A fitted model object with attributes:
          - f.fn: callable model function
          - f.value: list/array of fitted parameter values
    x : np.array
        Independent variable values.
    y : np.array
        Dependent (observed) values.
    sy : np.array
        Uncertainties associated with `y`.

    Returns
    -------
    float
        Reduced chi-squared statistic of the fit.
    """
    return chi2f(f.fn, len(f.values), x, y, sy)


def gaussian(y, A, mu, sigma):
    return A * np.exp(-(y - mu)**2 / (2 * sigma**2))

def gaussianfl(y, A, mu, sigma, C):
    return A * np.exp(-(y - mu)**2 / (2 * sigma**2)) + C
    


def gaussian_profiler_y(counts, xedges, yedges, min_counts=10):
    
    """
    Extracts the Gaussian mean and its uncertainty from vertical slices of a 2D histogram.

    Parameters:
    -----------
    counts : 2D np.ndarray
        The 2D histogram array of counts, typically from `np.histogram2d`.
    xedges : 1D np.ndarray
        Bin edges along the x-axis (e.g., DT).
    yedges : 1D np.ndarray
        Bin edges along the y-axis (e.g., Zrms).
    min_counts : int, optional (default=10)
        Minimum total counts required in a vertical slice for a fit to be attempted.

    Returns:
    --------
    dt_centers : np.ndarray
        The x-axis (DT) bin centers where a valid Gaussian fit was performed.
    mean_values : np.ndarray
        The Gaussian mean (μ) extracted from each vertical slice.
    mean_errors : np.ndarray
        The estimated statistical error on the fitted μ, derived from the fit covariance matrix..
    sigmas : np.ndarray
    The Gaussian width (σ) extracted from each vertical slice.

    sigma_errors : np.ndarray
    The estimated error on the fitted σ, derived from the fit covariance matrix.
    """

    # Bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    mean_values = []
    mean_errors = []
    sigma_vals = []
    sigma_errors = []
    dt_centers = []

    for i in range(len(xcenters)):
        profile = counts[i, :]
        if np.sum(profile) < min_counts or np.all(profile == 0):
            continue

        try:
            popt, pcov = curve_fit(gaussian, ycenters, profile, p0=gauss_seed(ycenters,profile))
            _, mu_fit, sigma_fit = popt
            mu_err = np.sqrt(pcov[2, 2])
            sigma_err = np.sqrt(pcov[2, 2]) 
            
            dt_centers.append(xcenters[i])
            mean_values.append(mu_fit)
            mean_errors.append(mu_err)  # error on mean
            sigma_vals.append(sigma_fit)
            sigma_errors.append(sigma_err)
            
        except RuntimeError:
            continue

    return np.array(dt_centers), np.array(mean_values), np.array(mean_errors), np.array(sigma_vals), np.array(sigma_errors) 



#Reimplementation with slice numbers 
def gaussian_profiler_y_slices(counts, xedges, yedges, slices=10, min_counts=10):
    """
    Fit vertical slices of a 2D histogram with Gaussians along y-axis.

    Parameters:
        counts : 2D array
            The 2D histogram counts.
        xedges, yedges : arrays
            Bin edges along x and y axes.
        slices : int or array-like
            Number of vertical slices or explicit x bin edges.
        min_counts : int
            Minimum counts in a slice to attempt a fit.

    Returns:
        dt_centers, mean_values, mean_errors, sigma_vals, sigma_errors
    """

    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Determine slicing
    if isinstance(slices, int):
        # Uniform slicing over full x-range
        custom_xedges = np.linspace(xedges[0], xedges[-1], slices + 1)
    else:
        # Use custom array of bin edges
        custom_xedges = np.asarray(slices)

    dt_centers = []
    mean_values = []
    mean_errors = []
    sigma_vals = []
    sigma_errors = []

    for i in range(len(custom_xedges) - 1):
        xlow, xhigh = custom_xedges[i], custom_xedges[i + 1]
        # Find histogram bins overlapping this range
        xmask = (xedges[:-1] >= xlow) & (xedges[1:] <= xhigh)
        if not np.any(xmask):
            continue

        # Average counts over selected bins along x
        profile = np.mean(counts[xmask, :], axis=0)

        if np.sum(profile) < min_counts or np.all(profile == 0):
            continue

        try:
            popt, pcov = curve_fit(gaussian, ycenters, profile, p0=gauss_seed(ycenters, profile))
            _, mu_fit, sigma_fit = popt
            mu_err = np.sqrt(np.diag(pcov))[1] if pcov.shape == (3, 3) else np.nan
            sigma_err = np.sqrt(np.diag(pcov))[2] if pcov.shape == (3, 3) else np.nan

            center = 0.5 * (xlow + xhigh)
            dt_centers.append(center)
            mean_values.append(mu_fit)
            mean_errors.append(mu_err)
            sigma_vals.append(sigma_fit)
            sigma_errors.append(sigma_err)
            
        except RuntimeError:
            dt_centers.append(np.nan)
            mean_values.append(np.nan)
            mean_errors.append(np.nan)
            sigma_vals.append(np.nan)
            sigma_errors.append(np.nan)
            continue
        

    return (np.array(dt_centers),
            np.array(mean_values),
            np.array(mean_errors),
            np.array(sigma_vals),
            np.array(sigma_errors))



#Reimplementation with slice numbers 
def gaussianfl_profiler_y_slices(counts, xedges, yedges, slices=10, min_counts=10):
    """
    Fit vertical slices of a 2D histogram with Gaussians+ a flat term along y-axis.

    Parameters:
        counts : 2D array
            The 2D histogram counts.
        xedges, yedges : arrays
            Bin edges along x and y axes.
        slices : int or array-like
            Number of vertical slices or explicit x bin edges.
        min_counts : int
            Minimum counts in a slice to attempt a fit.

    Returns:
        dt_centers, mean_values, mean_errors, sigma_vals, sigma_errors
    """

    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Determine slicing
    if isinstance(slices, int):
        # Uniform slicing over full x-range
        custom_xedges = np.linspace(xedges[0], xedges[-1], slices + 1)
    else:
        # Use custom array of bin edges
        custom_xedges = np.asarray(slices)

    dt_centers = []
    mean_values = []
    mean_errors = []
    sigma_vals = []
    sigma_errors = []

    for i in range(len(custom_xedges) - 1):
        xlow, xhigh = custom_xedges[i], custom_xedges[i + 1]
        # Find histogram bins overlapping this range
        xmask = (xedges[:-1] >= xlow) & (xedges[1:] <= xhigh)
        if not np.any(xmask):
            continue

        # Average counts over selected bins along x
        profile = np.mean(counts[xmask, :], axis=0)

        if np.sum(profile) < min_counts or np.all(profile == 0):
            continue

        try:
            popt, pcov = curve_fit(gaussianfl, ycenters, profile, p0=gaussfl_seed(ycenters, profile))
            _, mu_fit, sigma_fit, _ = popt
            mu_err = np.sqrt(np.diag(pcov))[1] if pcov.shape == (4, 4) else np.nan
            sigma_err = np.sqrt(np.diag(pcov))[2] if pcov.shape == (4, 4) else np.nan

            center = 0.5 * (xlow + xhigh)
            dt_centers.append(center)
            mean_values.append(mu_fit)
            mean_errors.append(mu_err)
            sigma_vals.append(sigma_fit)
            sigma_errors.append(sigma_err)
            
        except RuntimeError:
            dt_centers.append(np.nan)
            mean_values.append(np.nan)
            mean_errors.append(np.nan)
            sigma_vals.append(np.nan)
            sigma_errors.append(np.nan)
            continue
        

    return (np.array(dt_centers),
            np.array(mean_values),
            np.array(mean_errors),
            np.array(sigma_vals),
            np.array(sigma_errors))





def response_in_sector(df, sector_angle, radial_bins_per_sector, radius=480, center=(0, 0), dpi=180):
    """
    Compute mean and sigma of response in polar sectors and radial bins.

    Parameters:
        df : pd.DataFrame
            Input dataframe containing 'X', 'Y', and 'S2' (or relevant response column).
        sector_angle : float
            Angular width of each sector in degrees (e.g., 60 → 6 sectors).
        radial_bins_per_sector : int
            Number of radial divisions in each sector.
        radius : float
            Maximum radius of the circular region.
        center : tuple
            Center of the circle (default: (0, 0)).
        dpi : int
            Unused for now, just carried from original signature.

    Returns:
        results : np.ndarray
            Shape (n_sectors, n_radial_bins, 4) with:
            [mean, sigma, mean_error, sigma_error] for each sector and radial bin.
        radial_centers : np.ndarray
            Centers of radial bins.
    """
    
    df_radial = df.copy()

    # Convert to polar coordinates
    df_radial['theta_rad'] = np.arctan2(df_radial['Y'] - center[1], df_radial['X'] - center[0])
    df_radial['theta_deg'] = np.degrees(df_radial['theta_rad']) % 360
    df_radial['slice'] = (df_radial['theta_deg'] // sector_angle).astype(int)
    df_radial['R'] = np.sqrt((df_radial['X'] - center[0])**2 + (df_radial['Y'] - center[1])**2)

    # Radial bins
    radial_bins = np.linspace(0, radius, radial_bins_per_sector + 1)
    df_radial['radial_bin'] = np.digitize(df_radial['R'], radial_bins) - 1
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    
    assert 360 % sector_angle == 0, "sector_angle must divide 360 evenly"
    n_sectors = 360 // sector_angle
        
    results = np.full((n_sectors, radial_bins_per_sector, 4), np.nan)

    for sector_idx in range(n_sectors):
        
        bin_data = df_radial[
            (df_radial['slice'] == sector_idx) 
        ]

        hist, xedges, yedges = np.histogram2d(
            bin_data["R"], bin_data["Epes"],
            bins=(100, 120),
            range=[ (0,radius), (6000,10000) ]
        )

        r_centers, mean_vals, mean_errs, sigma_vals, sigma_errs = gaussian_profiler_y_slices(
                hist, xedges, yedges, slices=radial_bins_per_sector
        )

        for radial_idx in range(len(r_centers)):  # or radial_bins_per_sector if all bins are filled
            results[sector_idx, radial_idx] = [
                mean_vals[radial_idx],
                mean_errs[radial_idx],
                sigma_vals[radial_idx],
                sigma_errs[radial_idx]
            ]
        
    return results, r_centers


def plot_response_vs_radius_insector(r_centers, results, sector_angle,compute_resolution_with_error,color_sequence):
    """
    Plot Gaussian mean and resolution vs radius for each angular sector.

    Parameters:
        r_centers: 1D array
            Centers of radial bins.
        results: 3D array
            Array of shape (n_sectors, n_radial_bins, 4): mean, sigma, mean_err, sigma_err.
        sector_angle: int
            Angular width of each sector in degrees.
    """

    n_sectors = 360 // sector_angle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=120, sharex=True)

    for slice_idx in range(n_sectors):
        sector_data = results[slice_idx]

        mean = sector_data[:, 0]
        mean_err = sector_data[:, 1]
        sigma = sector_data[:, 2]
        sigma_err = sector_data[:, 3]

        resolution, resolution_err = compute_resolution_with_error(sigma, sigma_err, mean, mean_err)

        label = f'Slice {slice_idx*sector_angle}-{(slice_idx+1)*sector_angle}°'
        color = color_sequence[slice_idx % len(color_sequence)]

        ax1.errorbar(
            r_centers,
            mean,
            yerr=mean_err,
            fmt='o',
            markersize=6,
            color=color,
            label=label
        )

        ax2.errorbar(
            r_centers,
            resolution,
            yerr=resolution_err,
            fmt='o',
            markersize=6,
            color=color,
            label=label
        )

    ax1.set_ylabel('S2e (pes)')
    ax1.set_xlabel("Radius (mm)")
    ax1.legend(fontsize=10)
    ax1.grid(True)
    ax1.set_ylim(mean.mean() - 30, mean.mean() + 30)

    ax2.set_ylabel('Energy Resolution (%, FWHM)')
    ax2.set_xlabel("Radius (mm)")
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.set_ylim(2, 8)

    plt.tight_layout()
    plt.show()



def save_kr_map(mappar: MapPar, filename: str):
    np.savez_compressed(
        filename,
        hratio=mappar.hratio,
        hcounts=mappar.hcounts,
        xedges=mappar.xedges,
        yedges=mappar.yedges,
        zedges=mappar.zedges,
        hmap=mappar.hmap if mappar.hmap is not None else np.array([])
    )

def load_kr_map(filename: str) -> MapPar:
    data = np.load(filename)
    hmap_loaded = data['hmap']
    return MapPar(
        hratio=data['hratio'],
        hcounts=data['hcounts'],
        xedges=data['xedges'],
        yedges=data['yedges'],
        zedges=data['zedges'],
        hmap=hmap_loaded if hmap_loaded.size > 0 else None
    )


