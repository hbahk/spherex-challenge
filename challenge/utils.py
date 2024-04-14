# -*- coding: utf-8 -*-

# Author: Hyeonguk Bahk (bahkhyeonguk@gmail.com)
# Date: 2024-04-09 Tue
# Brief: Photometric redshift estimation for simulated SPHEREx data using EAZY
# All KASI codes were written by Dr. Sungryong Hong and reformatted in black style.

from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from eazy import filters, utils


def make_eazy_filters_spherex(filtdir, out, Nchan=17, Ndet=6,
                              path_default_filter=os.path.join(utils.path_to_eazy_data(), 'FILTER.RES.latest')):
    
    latest_filters = filters.FilterFile(path_default_filter)
    tempfile = Path('tempfile')
    with open(tempfile, 'w') as f:
        for j in range(Ndet):
            for i in range(Nchan):
                filt = Table.read(filtdir/f'filt_det{j+1}_{i+1}.csv',
                                  format='ascii.no_header',
                                  names=['wave', 'throughput'])
                wmask = filt['throughput'] > 1e-5
                res = filters.FilterDefinition(wave=filt['wave'][wmask],
                                            throughput=filt['throughput'][wmask],
                                            name=f'SPHEREx_Band{j+1}_{i+1}')
                f.write(res.for_filter_file() + '\n')

    temp_filters = filters.FilterFile('tempfile')
    tempfile.unlink()
    
    for filt in temp_filters.filters:
        latest_filters.filters.append(filt)
        
    latest_filters.write(out)
    return latest_filters


def flam_to_fnu(lam, flam):
    """
    Converts flux density per unit wavelength (flam) to flux density per unit
    frequency (fnu).

    Parameters:
    lam (float): Wavelength in microns.
    flam (float): Flux density per unit wavelength in Jy.

    Returns:
    fnu (float): Flux density per unit frequency in uJy.
    """
    fnu = 3.34e4 * lam**2 * flam  # in Jy
    return fnu * 1e6  # in uJy


def fnu_to_flam(lam, fnu):
    """
    Convert flux density in uJy to flux density in erg/s/cm^2/A.

    Parameters:
    lam (float): Wavelength in Angstroms.
    fnu (float): Flux density in microJanskys (uJy).

    Returns:
    float: Flux density in erg/s/cm^2/A.
    """
    flam = fnu / (3.34e4*lam**2) # fnu in uJy
    return flam * 1e-6 # in erg/s/cm^2/A

def fnu_to_abmag(fnu):
    """
    Converts flux density in microjanskys (uJy) to AB magnitude.

    Parameters:
    fnu (float): Flux density in microjanskys (uJy).

    Returns:
    float: AB magnitude.

    """
    return -2.5*np.log10(fnu) + 23.9 # fnu in uJy

def abmag_to_fnu(abmag):
    """
    Convert AB magnitude to flux density in microjansky (uJy).

    Parameters:
    abmag (float): The AB magnitude to be converted.

    Returns:
    float: The flux density in microjansky (uJy).
    """
    return 10**(-0.4*(abmag - 23.9)) # in uJy


def plot_comp_hexbin(
    z_spec,
    z_phot,
    z_phot_chi2,
    out,
    ids,
    label_x="",
    label_y="",
    title="",
    xmin=0.03,
    xmax=30.0,
    cmap="viridis",
    gridsize=(50, 50),
    scatter_plot=False,
    log_scale=True,
):
    """
    Plots a hexbin plot comparing spectroscopic redshifts (z_spec) and photometric
    redshifts (z_phot).

    Parameters:
    - z_spec (array-like): Array of spectroscopic redshifts.
    - z_phot (array-like): Array of photometric redshifts.
    - z_phot_chi2 (array-like): Array of chi-squared values for photometric redshifts.
    - out (str): Output file path for saving the plot.
    - ids (array-like): Array of object IDs.
    - label_x (str, optional): Label for the x-axis. Default is an empty string.
    - label_y (str, optional): Label for the y-axis. Default is an empty string.
    - title (str, optional): Title for the plot. Default is an empty string.
    - xmin (float, optional): Minimum value for the x-axis. Default is 0.03.
    - xmax (float, optional): Maximum value for the x-axis. Default is 30.0.
    - cmap (str, optional): Colormap for the hexbin plot. Default is "viridis".
    - scatter_plot (bool, optional): Whether to include a scatter plot of the data
        points. Default is False.

    Returns:
    - outlier (array-like): Array of object IDs for the outliers.

    """

    z_cnd = (z_phot > 0.0) & (z_spec > 0.0) & (z_phot_chi2 > 0.0)
    print(f"Objects : {np.sum(z_cnd):d}")

    delta_z = z_spec - z_phot
    dz = delta_z / (1 + z_spec)
    bias = np.mean(dz[z_cnd])
    # Normalized Median Absolute Deviation (NMAD)
    nmad = 1.48 * np.median(
        np.abs(delta_z[z_cnd] - np.median(delta_z[z_cnd])) / (1 + z_spec[z_cnd])
    )
    sigma = np.std(dz[z_cnd])
    

    outlier = z_cnd & (np.abs(dz) >= 0.15)
    print(f"Outliers: {np.sum(outlier):d}")
    print("\n")

    fig, ax = plt.subplots(figsize=(12, 10))

    if log_scale:
        logxmin, logxmax = np.log10(xmin), np.log10(xmax)

        hb = ax.hexbin(
            np.log10(z_spec[z_cnd]),
            np.log10(z_phot[z_cnd]),
            gridsize=gridsize,
            cmap=cmap,
            bins="log",
        )

        ax.plot([logxmin, logxmax], [logxmin, logxmax], "-", lw=1, color="k", alpha=0.75)
        if scatter_plot:
            ax.scatter(
                np.log10(z_spec[z_cnd]), np.log10(z_phot[z_cnd]), c="k", s=0.5, alpha=0.5
            )
        xx = np.logspace(logxmin, logxmax, 1000)
        logxx = np.log10(xx)
        logyy_lower = np.log10((1.0 - 0.15) * xx - 0.15)
        logyy_upper = np.log10((1.0 + 0.15) * xx + 0.15)
        ax.plot(logxx, logyy_lower, "--", lw=1, color="k", alpha=0.7)
        ax.plot(logxx, logyy_upper, "--", lw=1, color="k", alpha=0.7)
        
        ticks = np.array([1e-2, 1e-1, 1, 2, 3, 4, 5, 6])
        logticks = np.log10(ticks)
        ax.set_xticks(logticks)
        ax.set_yticks(logticks)
        ax.set_xticklabels([f"{x:.2f}" if x < 1 else f"{int(x)}" for x in ticks])
        ax.set_yticklabels([f"{x:.2f}" if x < 1 else f"{int(x)}" for x in ticks])
        ax.set_xlim([logxmin, logxmax])
        ax.set_ylim([logxmin, logxmax])
    else:
        hb = ax.hexbin(
            z_spec[z_cnd], z_phot[z_cnd], gridsize=gridsize, cmap=cmap, bins="log",
            extent=[xmin, xmax, xmin, xmax]
        )

        ax.plot([xmin, xmax], [xmin, xmax], "-", lw=1, color="k", alpha=0.75)
        if scatter_plot:
            ax.scatter(z_spec[z_cnd], z_phot[z_cnd], c="k", s=0.5, alpha=0.5)
        xx = np.linspace(xmin, xmax, 1000)
        ax.plot(xx, (1.0 - 0.15) * xx - 0.15, "--", lw=1, color="k", alpha=0.7)
        ax.plot(xx, (1.0 + 0.15) * xx + 0.15, "--", lw=1, color="k", alpha=0.7)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.tick_params(axis="both")
    ax.tick_params(width=1.5, length=7.5)
    ax.tick_params(width=1.5, length=4.0, which="minor")
    
    ax.text(
        0.95,
        0.05,
        r"$N$"
        + f" = {np.sum(z_cnd):d}\n"
        + r"$N_{\rm out}$"
        + f" = {np.sum(outlier):d}\n"
        + r"$\eta$"
        + f" = {100.*np.sum(outlier)/np.sum(z_cnd):.1f}%\n"
        + r"$\sigma_{\rm NMAD}$"
        + f" = {nmad:.3f}\n"
        + r"$\sigma$"
        + f" = {sigma:.3f}\n"
        + f"bias = {bias:.3f}",
        fontsize=18.0,
        color="black",
        bbox=dict(
            facecolor="white", boxstyle="round,pad=0.5", edgecolor="k", alpha=0.8,
        ),
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_title(title)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(hb, cax=cax, label="counts")
    ax.set_aspect("equal")

    # cb = plt.colorbar(hb, ax=ax, label="counts")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    #     plt.close()

    return ids[outlier]


def kasi_fit_object(ez, iobj=0, z=0):
    """
    Fit on the redshift grid
    """
    from scipy.optimize import nnls

    # import np.linalg

    fnu_i = ez.fnu[iobj, :] * ez.ext_redden
    efnu_i = ez.efnu[iobj, :] * ez.ext_redden
    ok_band = (fnu_i > ez.param["NOT_OBS_THRESHOLD"]) & (efnu_i > 0)

    A = ez.tempfilt(z)
    var = (0.0 * fnu_i) ** 2 + efnu_i**2 + (ez.TEF(z) * fnu_i) ** 2

    chi2 = np.zeros(ez.NZ, dtype=ez.ARRAY_DTYPE)
    coeffs = np.zeros((ez.NZ, ez.NTEMP), dtype=ez.ARRAY_DTYPE)

    for iz in range(ez.NZ):
        A = ez.tempfilt(ez.zgrid[iz])
        var = (0.0 * fnu_i) ** 2 + efnu_i**2 + (ez.TEF(ez.zgrid[iz]) * fnu_i) ** 2
        rms = np.sqrt(var)

        ok_temp = np.sum(A, axis=1) > 0
        if ok_temp.sum() == 0:
            chi2[iz] = np.inf
            coeffs[iz, :] = 0.0
            continue

        try:
            coeffs_x, rnorm = nnls(
                (A / rms).T[ok_band, :][:, ok_temp], (fnu_i / rms)[ok_band]
            )
            coeffs_i = np.zeros(A.shape[0], dtype=ez.ARRAY_DTYPE)
            coeffs_i[ok_temp] = coeffs_x
        except:
            coeffs_i = np.zeros(A.shape[0], dtype=ez.ARRAY_DTYPE)

        fmodel = np.dot(coeffs_i, A)
        chi2[iz] = np.sum((fnu_i - fmodel) ** 2 / var * ok_band)
        coeffs[iz, :] = coeffs_i

    return iobj, chi2, coeffs, fmodel


def kasi_compute_lnp(
    ez, prior=False, beta_prior=False, clip_wavelength=1100, in_place=True
):

    import time

    has_chi2 = (ez.chi2_fit != 0).sum(axis=1) > 0
    # min_chi2 = ez.chi2_fit[has_chi2,:].min(axis=1)

    loglike = -ez.chi2_fit[has_chi2, :] / 2.0
    # pz = np.exp(-(ez.chi2_fit[has_chi2,:].T-min_chi2)/2.).T

    if ez.param["VERBOSITY"] >= 2:
        print("compute_lnp ({0})".format(time.ctime()))

    if hasattr(ez, "tef_lnp"):
        if ez.param["VERBOSITY"] >= 2:
            print(" ... tef_lnp")

        loglike += ez.tef_lnp[has_chi2, :]

    if prior:
        if ez.param["VERBOSITY"] >= 2:
            print(" ... full_logprior")

        loglike += ez.full_logprior[has_chi2, :]

    if clip_wavelength is not None:
        # Set pz=0 at redshifts where clip_wavelength beyond reddest
        # filter
        clip_wz = clip_wavelength * (1 + ez.zgrid)
        red_mask = (clip_wz[:, None] > ez.lc_reddest[None, has_chi2]).T

        loglike[red_mask] = -np.inf
        ez.lc_zmax = ez.lc_reddest / clip_wavelength - 1
        ez.clip_wavelength = clip_wavelength

    if beta_prior:
        if ez.param["VERBOSITY"] >= 2:
            print(" ... beta lnp_beta")

        p_beta = ez.prior_beta(w1=1350, w2=1800, sample=has_chi2)
        ez.lnp_beta[has_chi2, :] = np.log(p_beta)
        ez.lnp_beta[~np.isfinite(ez.lnp_beta)] = -np.inf
        loglike += ez.lnp_beta[has_chi2, :]

    # Optional extra prior
    if hasattr(ez, "extra_lnp"):
        loglike += ez.extra_lnp[has_chi2, :]

    loglike[~np.isfinite(loglike)] = -1e20

    lnpmax = loglike.max(axis=1)
    pz = np.exp(loglike.T - lnpmax).T
    log_norm = np.log(pz.dot(ez.trdz))

    lnp = (loglike.T - lnpmax - log_norm).T
    # lnpmax = -log_norm

    lnp[~np.isfinite(lnp)] = -1e20

    if in_place:
        ez.lnp[has_chi2, :] = lnp
        ez.lnpmax[has_chi2] = -log_norm

        ez.lnp_with_prior = prior
        ez.lnp_with_beta_prior = beta_prior
    else:
        return has_chi2, lnp, -log_norm

    del lnpmax
    del pz
    del log_norm
    del loglike
    del lnp


def kasi_compute_lnp_object(
    idx_fit, in_chi2_fit_object, ez, prior=False, beta_prior=False, clip_wavelength=1100
):
    """
    The main issue : `idx_fit` was an 1 dim array in the original code.
    In this modding code, it should a single value, not an array.
    Hence, for the compatibility, each single value should be wrapped as a one-element 1 D array,
    such as [0] instead of 0.

    So.. loglike should be wrapped as loglike[None,:] to forcefully make it as a multi-dim array.

    The final output also should lower the array dimension such as lnp[0] and lnpmax[0] from 2d to 1d.
    """

    import time

    has_chi2 = idx_fit  # now this func will fit each object
    # has_chi2 = (ez.chi2_fit != 0).sum(axis=1) > 0
    # min_chi2 = ez.chi2_fit[has_chi2,:].min(axis=1)

    loglike = -1.0 * in_chi2_fit_object / 2.0
    # loglike = -ez.chi2_fit[has_chi2,:]/2.
    # pz = np.exp(-(ez.chi2_fit[has_chi2,:].T-min_chi2)/2.).T

    if ez.param["VERBOSITY"] >= 2:
        print("compute_lnp ({0})".format(time.ctime()))

    if hasattr(ez, "tef_lnp"):
        if ez.param["VERBOSITY"] >= 2:
            print(" ... tef_lnp")

        loglike += ez.tef_lnp[has_chi2, :]

    if prior:
        if ez.param["VERBOSITY"] >= 2:
            print(" ... full_logprior")

        loglike += ez.full_logprior[has_chi2, :]

    if clip_wavelength is not None:
        # Set pz=0 at redshifts where clip_wavelength beyond reddest
        # filter
        clip_wz = clip_wavelength * (1 + ez.zgrid)
        red_mask = (clip_wz[:, None] > ez.lc_reddest[None, has_chi2]).T

        ## potential "bug"
        # for each object, red_mask should be a single-dim array.
        # loglike[red_mask] = -np.inf
        loglike[red_mask[0]] = -np.inf
        ez.lc_zmax = ez.lc_reddest / clip_wavelength - 1
        ez.clip_wavelength = clip_wavelength

    if beta_prior:
        if ez.param["VERBOSITY"] >= 2:
            print(" ... beta lnp_beta")

        p_beta = ez.prior_beta(w1=1350, w2=1800, sample=has_chi2)
        ez.lnp_beta[has_chi2, :] = np.log(p_beta)
        ez.lnp_beta[~np.isfinite(ez.lnp_beta)] = -np.inf
        loglike += ez.lnp_beta[has_chi2, :]

    # Optional extra prior
    if hasattr(ez, "extra_lnp"):
        loglike += ez.extra_lnp[has_chi2, :]

    loglike[~np.isfinite(loglike)] = -1e20

    # print("len(loglike.shape) = "+str(len(loglike.shape)))
    if len(loglike.shape) != 1:
        print("Something is wrong in kasi_compute_lnp_object.")
        print("len(loglike.shape) = " + str(len(loglike.shape)))

    lnpmax = loglike[None, :].max(axis=1)
    pz = np.exp(loglike[None, :].T - lnpmax).T
    log_norm = np.log(pz.dot(ez.trdz))

    lnp = (loglike[None, :].T - lnpmax - log_norm).T
    lnp[~np.isfinite(lnp)] = -1e20

    # return has_chi2, lnp, -log_norm
    return has_chi2, lnp[0], -log_norm[0]

    del lnpmax
    del pz
    del log_norm
    del loglike
    del lnp


def kasi_get_redshift_object(
    idx_fit,
    ez,
    templnp,
    templnpmax,
    get_best_fit=True,
    prior=True,
    beta_prior=True,
    clip_wavelength=1100,
):
    """Fit parabola to ``lnp`` to get best maximum"""
    # from scipy import polyfit, polyval
    from numpy import polyfit, polyval

    """
        # for `chi2` values for `zgrid`
        idummy, tempchi, tempcoeff, tempfmodel = \
        kasi_fit_object(ez,iobj=idx_fit,z=0)
        
        idummy2, templnp, templnpmax = \
        kasi_compute_lnp_object(idx_fit,tempchi,ez, prior=prior, \
                                beta_prior=beta_prior, clip_wavelength=clip_wavelength):
        """
    # instead of the above, we only need `templnp` and `templnpmax`
    # Hence, we add two arguments of `templnp` and `templnpmax` to the function.

    # self.compute_lnp(prior=prior, beta_prior=beta_prior,
    #                 clip_wavelength=clip_wavelength)

    # A single object, idx_fit, for z-fit
    # has_chi2 = idx_fit
    # Objects that have been fit
    # has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0

    # izbest0 = np.argmin(self.chi2_fit, axis=1)
    # izmax = np.argmax(self.lnp, axis=1)*has_chi2
    izmax = np.argmax(templnp)

    # Default return-values when Fit is not Possible
    zbest = ez.zgrid[izmax]
    lnpmax = templnpmax

    isFit = (izmax > 0) & (
        izmax < ez.NZ - 2
    )  # for parabola fit, we need 4 points at izmax-1, ... , izmax+2
    # mask = (izmax > 0) & (izmax < self.NZ-1) & has_chi2 # Old mask

    if isFit & get_best_fit:
        c = polyfit(ez.zgrid[izmax - 1 : izmax + 2], templnp[izmax - 1 : izmax + 2], 2)
        zbest = -c[1] / (2 * c[0])
        lnpmax = polyval(c, zbest)
    # for iobj in self.idx[mask]:
    #     iz = izmax[iobj]
    #
    #     c = polyfit(self.zgrid[iz-1:iz+2], self.lnp[iobj, iz-1:iz+2], 2)
    #
    #     zbest[iobj] = -c[1]/(2*c[0])
    #     lnpmax[iobj] = polyval(c, zbest[iobj])

    """
        #####
        # Analytic parabola fit
        iz_ = izmax[self.idx[mask]]
        
        _x = np.array([self.zgrid[iz-1:iz+2] for iz in iz_])
        _y = np.array([self.lnp[iobj, iz-1:iz+2] 
                       for iz, iobj in zip(iz_, self.idx[mask])])

        dx = np.diff(_x, axis=1).T
        dx2 = np.diff(_x**2, axis=1).T
        dy = np.diff(_y, axis=1).T

        c2 = (dy[1]/dx[1] - dy[0]/dx[0]) / (dx2[1]/dx[1] - dx2[0]/dx[0])
        c1 = (dy[0] - c2 * dx2[0])/dx[0]
        c0 = _y.T[0] - c1*_x.T[0] - c2*_x.T[0]**2
        
        _m = self.idx[mask]
        zbest[_m] = -c1/2/c2
        lnpmax[_m] = c2*zbest[_m]**2+c1*zbest[_m]+c0
        
        
        del(_x)
        del(_y)
        del(iz_)
        del(dx)
        del(dx2)
        del(dy)
        del(c2)
        del(c1)
        del(c0)
        del(_m)
        """

    return zbest, lnpmax


def kasi_wrapper_fit_each_object(
    idx_fit,
    ez,
    get_best_fit=True,
    prior=False,
    beta_prior=False,
    clip_wavelength=1100,
    fitter="nnls",
):

    import numpy as np

    # import matplotlib.pyplot as plt
    # import time
    # import multiprocessing as mp

    fnu_corr = ez.fnu[idx_fit, :] * ez.ext_redden * ez.zp
    efnu_corr = ez.efnu[idx_fit, :] * ez.ext_redden * ez.zp

    efnu_corr[ez.fnu[idx_fit, :] < ez.param["NOT_OBS_THRESHOLD"]] = (
        ez.param["NOT_OBS_THRESHOLD"] - 9.0
    )
    # t0 = time.time()

    idummy, tempchi, tempcoeff, tempfmodel = kasi_fit_object(ez, iobj=idx_fit, z=0)
    # print("idummy ="+str(idummy))

    """ We should avoid updating ez.DATAFIELDs to broad-cast `ez`. `ez` should only have setting parameters, not calculated results 
    ez.chi2_fit[idx_fit,:] = tempchi
    ez.fit_coeffs[idx_fit,:] = tempcoeff
    ez.fmodel[idx_fit,:] = tempfmodel
    kasi_compute_lnp(ez,prior=prior, beta_prior=beta_prior, in_place=True)
    """

    # now, templnp is an 1d array and templnpmax is a scalar
    idummy2, templnp, templnpmax = kasi_compute_lnp_object(
        idx_fit,
        tempchi,
        ez,
        prior=prior,
        beta_prior=beta_prior,
        clip_wavelength=clip_wavelength,
    )

    # Fit the z-best using Parabola function
    zbest, lnpmaxbest = kasi_get_redshift_object(
        idx_fit,
        ez,
        templnp,
        templnpmax,
        get_best_fit=get_best_fit,
        prior=prior,
        beta_prior=beta_prior,
        clip_wavelength=clip_wavelength,
    )

    # t1 = time.time()

    return [zbest, tempfmodel.tolist()]


def kasi_wrapper_fit_each_object_show_new(
    idx_fit,
    ez,
    outfile=False,
    showplot=True,
    verbose=False,
    get_best_fit=True,
    prior=False,
    beta_prior=False,
    clip_wavelength=1100,
    fitter="nnls",
):

    import numpy as np

    # import matplotlib.pyplot as plt
    import time

    # import multiprocessing as mp
    from collections import OrderedDict
    from eazy.photoz import template_lsq
    from eazy.photoz import utils
    from eazy.photoz import igm_module
    import astropy.units as u
    from scipy.integrate import cumtrapz

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # plot settings
    # plt.rc('font', family='serif')
    # plt.rc('font', serif='Times New Roman')
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["mathtext.fontset"] = "stix"

    # nearest, interp
    TEMPLATE_REDSHIFT_TYPE = "nearest"
    IGM_OBJECT = igm_module.Inoue14()

    fnu_corr = ez.fnu[idx_fit, :] * ez.ext_redden * ez.zp
    efnu_corr = ez.efnu[idx_fit, :] * ez.ext_redden * ez.zp

    efnu_corr[ez.fnu[idx_fit, :] < ez.param["NOT_OBS_THRESHOLD"]] = (
        ez.param["NOT_OBS_THRESHOLD"] - 9.0
    )

    # variables for show : SED
    fnu_show = np.squeeze(ez.fnu[idx_fit, :]) * ez.ext_redden * ez.zp
    efnu_show = np.squeeze(ez.efnu[idx_fit, :]) * ez.ext_redden * ez.zp
    ok_band_show = fnu_show / ez.zp > ez.param["NOT_OBS_THRESHOLD"]
    ok_band_show &= efnu_show / ez.zp > 0
    efnu_show[~ok_band_show] = ez.param["NOT_OBS_THRESHOLD"] - 9.0
    zspec_show = ez.ZSPEC[idx_fit]

    log_prior_show = ez.full_logprior[idx_fit, :].flatten()

    # print("ix: ",idx_fit)
    # print("id: ",ez.OBJID[idx_fit])
    # print("fnu: ",fnu_show)
    # print("efnu: ",efnu_show)
    # print("ok_band: ",ok_band_show)
    # print("zspec: ",zspec_show)

    t0 = time.time()

    idummy, tempchi, tempcoeff, tempfmodel = kasi_fit_object(ez, iobj=idx_fit, z=0)
    # print("idummy ="+str(idummy))

    """ We should avoid updating ez.DATAFIELDs to broad-cast `ez`. `ez` should only have setting parameters, not calculated results 
    ez.chi2_fit[idx_fit,:] = tempchi
    ez.fit_coeffs[idx_fit,:] = tempcoeff
    ez.fmodel[idx_fit,:] = tempfmodel
    kasi_compute_lnp(ez,prior=prior, beta_prior=beta_prior, in_place=True)
    """

    # now, templnp is an 1d array and templnpmax is a scalar
    idummy2, templnp, templnpmax = kasi_compute_lnp_object(
        idx_fit,
        tempchi,
        ez,
        prior=prior,
        beta_prior=beta_prior,
        clip_wavelength=clip_wavelength,
    )

    # Fit the z-best using Parabola function
    zbest, lnpmaxbest = kasi_get_redshift_object(
        idx_fit,
        ez,
        templnp,
        templnpmax,
        get_best_fit=get_best_fit,
        prior=prior,
        beta_prior=beta_prior,
        clip_wavelength=clip_wavelength,
    )

    t1 = time.time()

    if verbose:
        print(">>Fitting time: {0:.3f} s ".format(t1 - t0))
        """
        print('>>idx_fit: '+str(idx_fit)+' lnpmax: '+str(ez.lnpmax[idx_fit]))
        """
        print(
            ">>idx_fit: "
            + str(idx_fit)
            + ", zbest: "
            + str(zbest)
            + ", lnpmax: "
            + str(lnpmaxbest)
        )

    # variables for show : Coeffs at zbest
    ndraws = ez.NDRAWS
    tef_show = ez.TEF(zbest)
    A = np.squeeze(ez.tempfilt(zbest))
    chi2_show, coeffs_show, fmodel, draws = template_lsq(
        fnu_show, efnu_show, A, tef_show, ez.zp, ndraws, fitter="nnls"
    )
    # print("ndraws: ",ndraws)
    # print("tef: ",tef_show)
    # print("chi2: ",chi2_show)
    # print("coeffs: ",coeffs_show)
    # print("fmodel: ",fmodel)
    # print("draws: ",draws)

    if draws is None:
        efmodel = 0
    else:
        efmodel = np.percentile(np.dot(draws, A), [16, 84], axis=0)
        efmodel = np.squeeze(np.diff(efmodel, axis=0) / 2.0)

    # print("efmodel",efmodel)

    # variables for show : Full SED
    templ = ez.templates[0]
    tempflux = np.zeros((ez.NTEMP, templ.wave.shape[0]), dtype=ez.ARRAY_DTYPE)

    for i in range(ez.NTEMP):
        zargs = {"z": zbest, "redshift_type": TEMPLATE_REDSHIFT_TYPE}
        fnu = ez.templates[i].flux_fnu(**zargs) * ez.tempfilt.scale[i]
    try:
        tempflux[i, :] = fnu
    except:
        tempflux[i, :] = np.interp(templ.wave, ez.templates[i].wave, fnu)

    templz = templ.wave * (1 + zbest)

    if ez.tempfilt.add_igm:
        igmz = templ.wave * 0.0 + 1
        lyman = templ.wave < 1300
        igmz[lyman] = IGM_OBJECT.full_IGM(zbest, templz[lyman])
    else:
        igmz = 1.0
    templf = np.dot(coeffs_show, tempflux) * igmz

    if draws is not None:
        templf_draws = np.dot(draws, tempflux) * igmz

    fnu_factor = 10 ** (-0.4 * (ez.param["PRIOR_ABZP"] + 48.6))

    templz_power = -2
    flam_spec = utils.CLIGHT * 1.0e10 / templz**2 / 1.0e-19
    flam_sed = utils.CLIGHT * 1.0e10 / ez.pivot**2 / ez.ext_corr / 1.0e-19
    ylabel = r"$f_\lambda [10^{-19}$ erg/s/cm$^2$]"
    flux_unit = 1.0e-19 * u.erg / u.s / u.cm**2 / u.AA

    # print("tempflux: ",tempflux)
    # print("igmz: ",igmz)

    try:
        showdata = OrderedDict(
            ix=idx_fit,
            idobj=ez.OBJID[idx_fit],
            zbest=zbest,
            z_spec=zspec_show,
            pivot=ez.pivot,
            model=fmodel * fnu_factor * flam_sed,
            emodel=efmodel * fnu_factor * flam_sed,
            fobs=fnu_show * fnu_factor * flam_sed,
            efobs=efnu_show * fnu_factor * flam_sed,
            valid=ok_band_show,
            tef=tef_show,
            templz=templz,
            templf=templf * fnu_factor * flam_spec,
            flux_unit=flux_unit,
            wave_unit=u.AA,
            chi2=chi2_show,
            coeffs=coeffs_show,
        )
    except:
        showdata = None

    print(">>Valid Bands: ", showdata["valid"])
    print(">>NumValid Bands: ", np.sum(showdata["valid"]))

    ###### Make the plot
    axes = None
    figsize = [12, 12]
    showpz = 0.4
    template_color = "#1f77b4"
    snr_thresh = 2.0
    with_tef = True
    show_upperlimits = True
    show_components = True
    show_redshift_draws = 200
    draws_cmap = None
    if showplot:

        if axes is None:
            fig = plt.figure(figsize=figsize)
            if showpz:
                fig_axes = GridSpec(2, 1, height_ratios=[1, showpz])
            else:
                fig_axes = GridSpec(1, 1, height_ratios=[1])

            ax = fig.add_subplot(fig_axes[0])
        else:
            fig = None
            fig_axes = None
            ax = axes[0]

        ax.scatter(
            showdata["pivot"] / 1.0e4,
            fmodel * fnu_factor * flam_sed,
            color="w",
            label=None,
            zorder=1,
            s=120,
            marker="o",
        )

        ax.scatter(
            showdata["pivot"] / 1.0e4,
            fmodel * fnu_factor * flam_sed,
            marker="x",
            color=template_color,
            label=None,
            zorder=2,
            s=120,
            alpha=0.8,
        )
        if draws is not None:
            ax.errorbar(
                showdata["pivot"] / 1.0e4,
                fmodel * fnu_factor * flam_sed,
                efmodel * fnu_factor * flam_sed,
                alpha=0.8,
                color=template_color,
                zorder=2,
                marker="None",
                linestyle="None",
                label=None,
            )

        # Missing data
        missing = fnu_show < ez.param["NOT_OBS_THRESHOLD"]
        missing |= efnu_show < 0

        # Detection
        sn2_detection = (~missing) & (fnu_show / efnu_show > snr_thresh)

        # S/N < 2
        sn2_not = (~missing) & (fnu_show / efnu_show <= snr_thresh)

        # Uncertainty with TEF
        if with_tef:
            err_tef = np.sqrt(efnu_show**2 + (tef_show * fnu_show) ** 2)
        else:
            err_tef = efnu_show * 1

        ax.errorbar(
            ez.pivot[sn2_detection] / 1.0e4,
            (fnu_show * fnu_factor * flam_sed)[sn2_detection],
            (err_tef * fnu_factor * flam_sed)[sn2_detection],
            color="k",
            marker="s",
            linestyle="None",
            label=None,
            zorder=10,
        )

        # show upper limits
        if show_upperlimits:
            ax.errorbar(
                ez.pivot[sn2_not] / 1.0e4,
                (fnu_show * fnu_factor * flam_sed)[sn2_not],
                (efnu_show * fnu_factor * flam_sed)[sn2_not],
                color="k",
                marker="s",
                alpha=0.4,
                linestyle="None",
                label=None,
            )

        pl = ax.plot(
            templz / 1.0e4,
            templf * fnu_factor * flam_spec,
            alpha=0.5,
            zorder=-1,
            color=template_color,
            label="z={0:.2f}".format(zbest),
        )

        # show components
        if show_components:
            colors = [
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

            for i in range(ez.NTEMP):
                if coeffs_show[i] != 0:
                    pi = ax.plot(
                        templz / 1.0e4,
                        coeffs_show[i] * tempflux[i, :] * igmz * fnu_factor * flam_spec,
                        alpha=0.5,
                        zorder=-1,
                        label=ez.templates[i].name.split(".dat")[0],
                        color=colors[i % len(colors)],
                    )

        if show_redshift_draws:

            if draws_cmap is None:
                draws_cmap = plt.cm.rainbow

            # Draw random values from p(z)
            pz = np.exp(templnp).flatten()
            pzcum = cumtrapz(pz, x=ez.zgrid)

            if show_redshift_draws == 1:
                nzdraw = 100
            else:
                nzdraw = show_redshift_draws * 1

            rvs = np.random.rand(nzdraw)
            zdraws = np.interp(rvs, pzcum, ez.zgrid[1:])

            for zi in zdraws:
                Az = np.squeeze(ez.tempfilt(zi))
                chi2_zi, coeffs_zi, fmodelz, __ = template_lsq(
                    fnu_show, efnu_show, Az, ez.TEF(zi), ez.zp, 0, fitter="nnls"
                )

                c_i = np.interp(zi, ez.zgrid, np.arange(ez.NZ) / ez.NZ)

                templzi = templ.wave * (1 + zi)
                if ez.tempfilt.add_igm:
                    igmz = templ.wave * 0.0 + 1
                    lyman = templ.wave < 1300
                    igmz[lyman] = IGM_OBJECT.full_IGM(zi, templzi[lyman])
                else:
                    igmz = 1.0

                templfz = np.dot(coeffs_zi, tempflux) * igmz
                templfz *= flam_spec * (templz / templzi) ** templz_power

                plz = ax.plot(
                    templzi / 1.0e4,
                    templfz * fnu_factor,
                    alpha=np.maximum(0.1, 1.0 / nzdraw),
                    zorder=-1,
                    color=draws_cmap(c_i),
                )

        if draws is not None:
            templf_width = np.percentile(
                templf_draws * fnu_factor * flam_spec, [16, 84], axis=0
            )
            ax.fill_between(
                templz / 1.0e4,
                templf_width[0, :],
                templf_width[1, :],
                color=pl[0].get_color(),
                alpha=0.1,
                label=None,
            )

        # show x y labels
        add_label = True
        FNTSIZE = 20
        xlim = [0.3, 30]
        if axes is None:
            ax.set_ylabel(ylabel)

            if sn2_detection.sum() > 0:
                ymax = (fmodel * fnu_factor * flam_sed)[sn2_detection].max()
            else:
                ymax = (fmodel * fnu_factor * flam_sed).max()

            if np.isfinite(ymax):
                ax.set_ylim(-0.1 * ymax, 1.2 * ymax)

            ax.set_xlim(xlim)
            xt = np.array([0.1, 0.5, 1, 2, 4, 8, 24, 160, 500]) * 1.0e4

            ax.semilogx()

            valid_ticks = (xt > xlim[0] * 1.0e4) & (xt < xlim[1] * 1.0e4)
            if valid_ticks.sum() > 0:
                xt = xt[valid_ticks]
                ax.set_xticks(xt / 1.0e4)
                ax.set_xticklabels(xt / 1.0e4)

            ax.set_xlabel(r"$\lambda_\mathrm{obs}$")
            ax.grid()

            if add_label:
                txt = "{0}\nID={1}"
                txt = txt.format(
                    ez.param["MAIN_OUTPUT_FILE"], showdata["idobj"]
                )  # , self.prior_mag_cat[ix])

                ax.text(
                    0.95,
                    0.95,
                    txt,
                    ha="right",
                    va="top",
                    fontsize=FNTSIZE,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="w", alpha=0.5),
                    zorder=10,
                )

                ax.legend(fontsize=12, loc="center right")

        ## P(z)
        show_prior = False
        logpz = False
        zr = [0, 6]
        zshow = zbest
        bool_pzlabels = False
        if not showpz:
            return fig, showdata

        if axes is not None:
            if len(axes) == 1:
                return fig, showdata
            else:
                ax = axes[1]
        else:
            ax = fig.add_subplot(fig_axes[1])

        chi2 = np.squeeze(chi2_show)
        prior = np.exp(log_prior_show)
        # pz = np.exp(-(chi2-chi2.min())/2.)*prior
        # pz /= np.trapz(pz, self.zgrid)
        pz = np.exp(templnp).flatten()

        ax.plot(ez.zgrid, pz, color="orange", label=None)
        if show_prior:
            ax.plot(ez.zgrid, prior / prior.max() * pz.max(), color="g", label="prior")

        ax.fill_between(ez.zgrid, pz, pz * 0, color="yellow", alpha=0.5, label=None)
        if zspec_show > 0:
            ax.vlines(
                zspec_show,
                1.0e-5,
                pz.max() * 1.05,
                color="r",
                label="zspec={0:.3f}".format(zspec_show),
            )

        if zshow is not None:
            ax.vlines(
                zshow,
                1.0e-5,
                pz.max() * 1.05,
                color=template_color,
                label="z={0:.3f}".format(zshow),
                linestyles="dashed",
            )

        if axes is None:
            ax.set_ylim(0, pz.max() * 1.05)

            if logpz:
                ax.semilogy()
                ymax = np.minimum(ax.get_ylim()[1], 100)
                ax.set_ylim(1.0e-3 * ymax, 1.8 * ymax)

            if zr is None:
                ax.set_xlim(0, ez.zgrid[-1])
            else:
                ax.set_xlim(zr)

            ax.set_xlabel("z")
            ax.set_ylabel("p(z)")
            ax.grid()
            if not bool_pzlabels:
                ax.set_yticklabels([])

            fig_axes.tight_layout(fig, pad=0.5)

            if add_label & (zspec_show > 0):
                ax.legend(fontsize=7, loc="upper left")
        # Save or not
        if outfile:
            fig.savefig(outfile)
            currentdir = os.getcwd()
            print(">>" + outfile + " is saved on your current dirctory: " + currentdir)

    return showdata
