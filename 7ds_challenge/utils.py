from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_comp_hexbin(
    z_spec,
    z_phot,
    z_phot_chi2,
    out,
    ids,
    z_840=None,
    z_160=None,
    label_x="",
    label_y="",
    title="",
    xmin=0.03,
    xmax=30.0,
    cmap="viridis",
    gridsize=(50, 50),
    scatter_plot=False,
    no_hexbin=False,
    log_scale=True,
    color_log_scale=True,
    residual_plot=False,
    residual_ylabel=r"$\Delta z / (1+z)$",
    residual_ylim=None,
    figsize=(12, 13),
    rfigsize=(12, 5),
    figclose=True,
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
    - z_840 (array-like, optional): Array of 84th percentile redshifts. Default is None.
    - z_160 (array-like, optional): Array of 16th percentile redshifts. Default is None.
    - label_x (str, optional): Label for the x-axis. Default is an empty string.
    - label_y (str, optional): Label for the y-axis. Default is an empty string.
    - title (str, optional): Title for the plot. Default is an empty string.
    - xmin (float, optional): Minimum value for the x-axis. Default is 0.03.
    - xmax (float, optional): Maximum value for the x-axis. Default is 30.0.
    - cmap (str, optional): Colormap for the hexbin plot. Default is "viridis".
    - scatter_plot (bool, optional): Whether to include a scatter plot of the data
        points. Default is False.
    - log_scale (bool, optional): Whether to use a log scale for the axes. Default is
        True.
    - color_log_scale (bool, optional): Whether to use a log scale for the color.
        Default is True.

    Returns:
    - outlier (array-like): Array of object IDs for the outliers.

    """

    z_cnd = (z_phot > 0.0) & (z_spec > 0.0) & (z_phot_chi2 > 0.0)
    print(f"Objects : {np.sum(z_cnd):d}")

    delta_z = z_phot - z_spec
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

    if z_840 is not None and z_160 is not None:
        sigz = (np.max([z_840 - z_phot, z_phot - z_160], axis=0) / (1 + z_phot))
        medsigz = np.median(sigz[z_cnd])

        outsigz = z_cnd & (np.abs(dz) > 3*sigz)
        outsigzfrac = np.sum(outsigz) / np.sum(z_cnd) * 100.

        text_append = (
            "\n"
            + r"$\tilde{\sigma}_{z/(1+z)}$" + f" = {medsigz:.3f}\n"
            + r"$\eta_{3\hat{\sigma}}$" + f" = {outsigzfrac:.1f}%"
        )
    else:
        text_append = ""

    if not residual_plot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize,
                               gridspec_kw={"height_ratios": [3, 1], "hspace":0})
        ax = axes[0]
        rax = axes[1]
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    bins = "log" if color_log_scale else None

    if log_scale:
        logxmin, logxmax = np.log10(1+xmin), np.log10(1+xmax)
        if not no_hexbin:
            hb = ax.hexbin(
                np.log10(1+z_spec[z_cnd]),
                np.log10(1+z_phot[z_cnd]),
                gridsize=gridsize,
                cmap=cmap,
                bins=bins,
                mincnt=1,
            )

        if scatter_plot:
            ax.scatter(
                np.log10(1+z_spec[z_cnd]), np.log10(1+z_phot[z_cnd]), c="k", s=0.5, alpha=0.5
            )
            if z_840 is not None and z_160 is not None:
                yerr = sigz[z_cnd]/(1+z_phot[z_cnd])
                ax.errorbar(np.log10(1+z_spec[z_cnd]), np.log10(1+z_phot[z_cnd]), yerr=yerr, fmt="o",
                            ms=3, lw=0.8, c="w", mec="k", ecolor="k")
            else:
                ax.scatter(np.log10(1+z_spec[z_cnd]), np.log10(1+z_phot[z_cnd]), c="w", s=10, lw=0.2, ec="k")
            
        ax.plot([logxmin, logxmax], [logxmin, logxmax], "-", lw=1, color="k", alpha=0.75)
        xx = np.logspace(logxmin, logxmax, 1000)
        logxx = np.log10(xx)
        logyy_lower = np.log10((1.0 - 0.15) * xx - 0.15)
        logyy_upper = np.log10((1.0 + 0.15) * xx + 0.15)
        ax.plot(logxx, logyy_lower, "--", lw=1, color="k", alpha=0.7)
        ax.plot(logxx, logyy_upper, "--", lw=1, color="k", alpha=0.7)

        ticks = np.array([1e-2, 1e-1, 1, 2, 3, 4, 5, 6])
        logticks = np.log10(1+ticks)
        ax.set_xticks(logticks)
        ax.set_yticks(logticks)
        ax.set_xticklabels([f"{x:.2f}" if x < 1 else f"{int(x)}" for x in ticks])
        ax.set_yticklabels([f"{x:.2f}" if x < 1 else f"{int(x)}" for x in ticks])
        ax.set_xlim([logxmin, logxmax])
        ax.set_ylim([logxmin, logxmax])
    else:
        if not no_hexbin:
            hb = ax.hexbin(
                z_spec[z_cnd], z_phot[z_cnd], gridsize=gridsize, cmap=cmap, bins=bins,
                extent=[xmin, xmax, xmin, xmax], mincnt=1
            )

        if scatter_plot:
            if z_840 is not None and z_160 is not None:
                color = np.abs(dz[z_cnd]/sigz[z_cnd])
                isc = np.argsort(color)[::-1]
                xsc, ysc = z_spec[z_cnd][isc], z_phot[z_cnd][isc]
                yesc = sigz[z_cnd][isc]
                colorsc = color[isc]
                ax.errorbar(xsc, ysc, yerr=yesc, fmt="o",
                            ms=5, lw=0.8, c="w", mec="k", ecolor="gray", zorder=1)
                csc = ax.scatter(xsc, ysc, c=colorsc, s=20, lw=0.2,
                                 ec="k", cmap="Purples_r", vmin=0, vmax=3, zorder=2)
            else:
                ax.scatter(z_spec[z_cnd], z_phot[z_cnd], c="w", s=20, lw=0.2, ec="k")
            
        ax.plot([xmin, xmax], [xmin, xmax], "-", lw=1, color="k", alpha=0.75)
        xx = np.linspace(xmin, xmax, 1000)
        ax.plot(xx, (1.0 - 0.15) * xx - 0.15, "--", lw=1, color="k", alpha=0.7)
        ax.plot(xx, (1.0 + 0.15) * xx + 0.15, "--", lw=1, color="k", alpha=0.7)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    ax.tick_params(axis="both")
    ax.tick_params(width=0.8, length=7.5)
    ax.tick_params(width=0.8, length=4.0, which="minor")

    ax.text(
        0.95,
        0.05,
        r"$N$"
        + f" = {np.sum(z_cnd):d}\n"
        + r"$N_{\rm out}$"
        + f" = {np.sum(outlier):d}\n"
        + r"$f_{0.15}$"
        + f" = {100.*np.sum(outlier)/np.sum(z_cnd):.1f}%\n"
        + r"$\sigma_{\rm NMAD}$"
        + f" = {nmad:.3f}\n"
        + r"$\sigma$"
        + f" = {sigma:.3f}\n"
        + f"bias = {bias:.3f}"
        + text_append,
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

    if not no_hexbin:
        cb = plt.colorbar(hb, cax=cax, label="counts")
    else:
        if z_840 is not None and z_160 is not None:
            cb = plt.colorbar(csc, cax=cax, label=r"$|\Delta z / (1+z)|$ [$\hat\sigma_{z/(1+\hat{z})}$]")

    if residual_plot:
        # rfig = plt.figure(figsize=rfigsize)
        # rax = rfig.add_subplot(111)
        ax.set_xticklabels([])
        
        if z_840 is not None and z_160 is not None:
            _yerr = sigz[z_cnd]
            idxs = np.argsort(_yerr)[::-1]
            yerr = _yerr[idxs]
        else:
            yerr = None
            idxs = np.full_like(z_spec[z_cnd], True)
            
        if log_scale:
            xx = np.log10(z_spec[z_cnd][idxs])
            rax.set_xlim([logxmin, logxmax])
        else:
            xx = z_spec[z_cnd][idxs]
            rax.set_xlim([xmin, xmax])
            
        
        rax.plot([xmin, xmax], [0, 0], "-", lw=0.8, color="k", alpha=0.75)
        rax.plot([xmin, xmax], [0.15, 0.15], "--", lw=0.8, color="k", alpha=0.75)
        rax.plot([xmin, xmax], [-0.15, -0.15], "--", lw=0.8, color="k", alpha=0.75)
        rsc = rax.scatter(xx, dz[z_cnd][idxs], c=yerr, s=20, zorder=2, lw=0.2, ec="k",
                          cmap="inferno_r", norm=LogNorm(),)
        rax.set_ylim(rax.get_ylim())
        rax.errorbar(xx, dz[z_cnd][idxs], yerr=yerr[idxs], fmt="o", ms=3, lw=0.8,
                     c="w", mec="k", ecolor="k", zorder=1)
        
        rax.set_xlabel(label_x)
        rax.set_ylabel(residual_ylabel)
        
        rdivider = make_axes_locatable(rax)
        rcax = rdivider.append_axes("right", size="5%", pad=0.1)
        rcb = plt.colorbar(rsc, cax=rcax, label=r"$\hat{\sigma}_{z/(1+\hat{z})}$")
        
        if residual_ylim:
            rax.set_ylim(residual_ylim)
            
            
    # ax.set_aspect("equal")
    ax.tick_params(axis="both", which="minor")
    ax.tick_params(axis="both", direction="in", which="both")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    if figclose:
        plt.close()

    return ids[outlier]


def get_result_figures(base, figdir, scheme):
    id_out = plot_comp_hexbin(base['z_true'], base['z_phot'], base['z_phot_chi2'],
                            figdir/'all_hexhist.png', base['id'],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title=f"EAZY 7DS ({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            color_log_scale=True, figsize=(12,11))
    
    imask = base['HSC_i_MAG'] < 19

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i19_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$i<19$"+f"({scheme})", xmin=0, xmax=0.9, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i19_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$i<19$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    imask = (base['HSC_i_MAG'] > 19) & (base['HSC_i_MAG'] < 20)

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i20_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$19<i<20$"+f"({scheme})", xmin=0.015, xmax=1.1, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i20_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$19<i<20$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    imask = (base['HSC_i_MAG'] > 20) & (base['HSC_i_MAG'] < 21)

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i21_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$20<i<21$"+f"({scheme})", xmin=0.015, xmax=1.5, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i21_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$20<i<21$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    imask = (base['HSC_i_MAG'] > 21) & (base['HSC_i_MAG'] < 22)

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i22_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$21<i<22$"+f"({scheme})", xmin=0.015, xmax=3, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i22_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$21<i<22$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    imask = (base['HSC_i_MAG'] > 22) & (base['HSC_i_MAG'] < 23)

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i23_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$22<i<23$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i23_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$22<i<23$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    imask = (base['HSC_i_MAG'] > 23)

    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i25_scatter.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$i>23$", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=True, gridsize=(87,50), log_scale=False,
                            residual_plot=True, no_hexbin=True)
    id_out = plot_comp_hexbin(base['z_true'][imask], base['z_phot'][imask],
                            base['z_phot_chi2'][imask],
                            figdir/'i25_hexhist.png', base['id'][imask],
                            z_160=base['z160'][imask], z_840=base['z840'][imask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title="$i>23$"+f"({scheme})", xmin=0.015, xmax=5.8, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)

    sigz = (
        np.max([base["z840"] - base["z_phot"], base["z_phot"] - base["z160"]], axis=0)
        / (1 + base["z_phot"])
    )
    idxs = np.argsort(sigz)[::-1]
    x, y, s = base['z_true'][idxs], base['z_phot'][idxs], sigz[idxs]
    sigzmask = (s > 0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x[sigzmask], y[sigzmask], c=s[sigzmask], s=1, cmap='inferno_r',
                    norm=LogNorm())
    ax.plot([0, 5.8], [0, 5.8], color='k', lw=0.8, ls='--')
    ax.set_xlim(0, 5.8)
    ax.set_ylim(0, 5.8)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_ylabel(r'$z_{\rm phot}$')
    ax.set_axisbelow(True)
    ax.grid()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(sc, cax=cax)
    cax.invert_yaxis()
    cbar.set_label(r'$\hat{\sigma}_{z/(1+\hat{z})}$')

    ax.set_title('Phot-z from EAZY'+f"({scheme})")
    fig.savefig(figdir/'sigz.png', dpi=300)
    plt.close()

    sigzmask = (s < 0.1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x[sigzmask], y[sigzmask], c=s[sigzmask], s=1, cmap='inferno_r',
                    norm=LogNorm())
    ax.plot([0, 5.8], [0, 5.8], color='k', lw=0.8, ls='--')
    ax.set_xlim(0, 5.8)
    ax.set_ylim(0, 5.8)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_ylabel(r'$z_{\rm phot}$')
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_title(r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.1$'+f"({scheme})")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(sc, cax=cax)
    cax.invert_yaxis()
    cbar.set_label(r'$\hat{\sigma}_{z/(1+\hat{z})}$')

    fig.savefig(figdir/'sigz_lt01.png', dpi=300)
    plt.close()


    sigzmask = (s < 0.01)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x[sigzmask], y[sigzmask], c=s[sigzmask], s=1, cmap='inferno_r',
                    norm=LogNorm())
    ax.plot([0, 5.8], [0, 5.8], color='k', lw=0.8, ls='--')
    ax.set_xlim(0, 5.8)
    ax.set_ylim(0, 5.8)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_ylabel(r'$z_{\rm phot}$')
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_title(r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.01$'+f"({scheme})")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(sc, cax=cax)
    cax.invert_yaxis()
    cbar.set_label(r'$\hat{\sigma}_{z/(1+\hat{z})}$')

    fig.savefig(figdir/'sigz_lt001.png', dpi=300)
    plt.close()


    sigzmask = (s < 0.1)

    id_out = plot_comp_hexbin(x[sigzmask], y[sigzmask],
                            np.ones_like(x[sigzmask]),
                            figdir/'sigz_hexbin_lt01.png', np.arange(len(x[sigzmask])),
                            z_840=base['z840'][idxs][sigzmask], z_160=base['z160'][idxs][sigzmask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title=r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.1$'+f"({scheme})",
                            xmin=0.015, xmax=2.5, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True)
    id_out = plot_comp_hexbin(x[sigzmask], y[sigzmask],
                            np.ones_like(x[sigzmask]),
                            figdir/'sigz_scatter_lt01.png', np.arange(len(x[sigzmask])),
                            z_840=base['z840'][idxs][sigzmask], z_160=base['z160'][idxs][sigzmask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title=r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.1$'+f"({scheme})",
                            xmin=0.015, xmax=2.5, cmap='jet',
                            scatter_plot=True, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True, no_hexbin=True)


    sigzmask = (s < 0.01)

    id_out = plot_comp_hexbin(x[sigzmask], y[sigzmask],
                            np.ones_like(x[sigzmask]),
                            figdir/'sigz_hexbin_lt001.png', np.arange(len(x[sigzmask])),
                            z_840=base['z840'][idxs][sigzmask], z_160=base['z160'][idxs][sigzmask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title=r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.01$'+f"({scheme})",
                            xmin=0.015, xmax=2.5, cmap='jet',
                            scatter_plot=False, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True, residual_ylim=None)
    id_out = plot_comp_hexbin(x[sigzmask], y[sigzmask],
                            np.ones_like(x[sigzmask]),
                            figdir/'sigz_scatter_lt001.png', np.arange(len(x[sigzmask])),
                            z_840=base['z840'][idxs][sigzmask], z_160=base['z160'][idxs][sigzmask],
                            label_x=r"$z_{\rm true}$", label_y=r"$z_{\rm phot}$",
                            title=r'$\hat{\sigma}_{z/(1+\hat{z})} < 0.01$'+f"({scheme})",
                            xmin=0.015, xmax=2.5, cmap='jet',
                            scatter_plot=True, gridsize=(87*2,50*2), log_scale=False,
                            residual_plot=True, residual_ylim=None, no_hexbin=True)