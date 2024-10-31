import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm
from eazy import utils
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

MULTIPROCESSING_TIMEOUT = 3600
MIN_VALID_FILTERS = 1


def fit_object_single_template(fnu_i, efnu_i, A, TEFz, zp, array_dtype):
    """
    Fit on the redshift grid
    """

    global MIN_VALID_FILTERS

    ntemp, nfilt = A.shape

    ok_band = (efnu_i / zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)

    var = (0.0 * fnu_i) ** 2 + efnu_i**2 + (TEFz * fnu_i) ** 2
    rms = np.sqrt(var)

    ok_temp = np.sum(A, axis=1) > 0
    if (ok_temp.sum() == 0) | (ok_band.sum() < MIN_VALID_FILTERS):
        chi2 = np.inf
        coeffs = np.zeros(ntemp)
    else:
        try:
            coeffs_x = np.zeros(ntemp, dtype=array_dtype)
            ampl_x_vec = np.zeros(ntemp, dtype=array_dtype)
            chi2_x_vec = np.zeros(ntemp)

            Atrans = A.T
            fnu_rms = fnu_i / zp / rms * ok_band
            rms_inv = 1.0 / rms * ok_band

            num = fnu_rms @ Atrans
            den = rms_inv @ (Atrans**2)

            ampl_x_vec = num / den
            mz_vec = ampl_x_vec[:, np.newaxis] * A
            mz_vec_trans = mz_vec.T
            chi2_x_vec = np.sum(
                ((mz_vec_trans - fnu_i[:, None] / zp[:, None]) * ok_band[:, None]) ** 2
                / var[:, None],
                axis=0,
            )

            iargmin = np.argmin(chi2_x_vec)
            coeffs_x[iargmin] = ampl_x_vec[iargmin]
            coeffs = np.zeros(ntemp, dtype=array_dtype)
            # coeffs[ok_temp] = coeffs_x
            coeffs = coeffs_x # TODO: this is a temporary fix; ok_temp is not used

        except:
            coeffs = np.zeros(ntemp, dtype=array_dtype)

    fmodel = np.dot(coeffs, A)
    chi2 = np.sum((fnu_i - fmodel) ** 2 / var * ok_band)

    return chi2, coeffs, fmodel


def fit_by_redshift_sigle_template(iz, z, A, fnu_corr, efnu_corr, TEFz, zp, verbose, array_dtype):
    NOBJ, NFILT = fnu_corr.shape
    NTEMP = A.shape[0]
    chi2 = np.zeros(NOBJ, dtype=fnu_corr.dtype)
    coeffs = np.zeros((NOBJ, NTEMP), dtype=fnu_corr.dtype)
    
    if verbose > 2:
        print('z={0:7.3f}'.format(z))
    
    for iobj in range(NOBJ):
        
        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]
        ok_band = (efnu_i > 0)
        
        if ok_band.sum() < 2:
            continue
        
        _res = fit_object_single_template(fnu_i, efnu_i, A, TEFz, zp, array_dtype)
        chi2[iobj], coeffs[iobj], fmodel = _res
            
    return iz, chi2, coeffs


def fit_at_zbest_single_template(self, zbest=None, prior=False, beta_prior=False, get_err=False, clip_wavelength=1100, fitter='nnls', selection=None,  n_proc=0, par_skip=10000, recompute_zml=True, **kwargs):
    """
    Recompute the fit coefficients at the "best" redshift.  
    
    If `zbest` not specified, then will fit at the maximum likelihood
    redshift from the `zml` attribute.
    
    """
    import multiprocessing as mp
            
    #izbest = np.argmin(self.chi2_fit, axis=1)
    izbest = self.izbest*1
    has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
    
    self.get_err = get_err
    
    self.zbest_grid = self.zgrid[izbest]
    #self.chi_best
    
    if zbest is None:
        if (self.zml is None):
            recompute_zml |= True
        else:
            # Recompute if prior options changed
            recompute_zml |= prior is not self.ZML_WITH_PRIOR
            recompute_zml |= beta_prior is not self.ZML_WITH_BETA_PRIOR
            
        if recompute_zml:
            self.evaluate_zml(prior=prior, beta_prior=beta_prior)

        self.ZPHOT_USER = False # user did *not* specify zbest
        self.zbest = self.zml
                        
    else:
        self.zbest = zbest
        self.ZPHOT_USER = True # user *did* specify zbest
                
    if ((self.param['FIX_ZSPEC'] in utils.TRUE_VALUES) & 
        ('z_spec' in self.cat.colnames)):
        has_zsp = self.ZSPEC > self.zgrid[0]
        self.zbest[has_zsp] = self.ZSPEC[has_zsp]
        self.ZPHOT_AT_ZSPEC = True
    else:
        self.ZPHOT_AT_ZSPEC = False
        
    # Compute Risk function at z=zbest
    self.zbest_risk = self.compute_best_risk()
    
    fnu_corr = self.fnu*self.ext_redden*self.zp
    efnu_corr = self.efnu*self.ext_redden*self.zp
    efnu_corr[~self.ok_data] = self.param['NOT_OBS_THRESHOLD'] - 9.
    
    subset = (self.zbest > self.zgrid[0]) & (self.zbest < self.zgrid[-1])
    if selection is not None:
        subset &= selection
        
    idx = self.idx[subset]
    
    # Set seed
    np.random.seed(self.random_seed)
    
    if n_proc <= 0:
        np_check = np.maximum(mp.cpu_count() - 2, 1)
    else:
        np_check = np.minimum(mp.cpu_count(), n_proc)
        
    # Fit in parallel mode        
    t0 = time.time()
    
    skip = np.maximum(len(idx)//par_skip, 1)
    np_check = np.minimum(np_check, skip)
    
    if get_err:
        get_err = self.NDRAWS
        
    if skip == 1:
        # Serial (pass self at end to update arrays in place)
        _ = _fit_at_zbest_group_single_template(idx, 
                            fnu_corr[idx,:], 
                            efnu_corr[idx,:], 
                            self.zbest[idx], 
                            self.zp*1, 
                            get_err,
                            self.tempfilt, 
                            self.TEF,
                            self.ARRAY_DTYPE, 
                            None)

        _ix, _coeffs_best, _fmodel, _efmodel, _chi2_best, _cdraws = _
        self.coeffs_best[_ix,:] = _coeffs_best
        self.fmodel[_ix,:] = _fmodel
        self.efmodel[_ix,:] = _efmodel
        self.chi2_best[_ix] = _chi2_best
        if get_err:
            self.coeffs_draws[_ix,:,:] = _cdraws
        
    else:
        # Multiprocessing
        pool = mp.Pool(processes=np_check)
        jobs = [pool.apply_async(_fit_at_zbest_group_single_template, 
                                        (idx[i::skip], 
                                        fnu_corr[idx[i::skip],:], 
                                        efnu_corr[idx[i::skip],:], 
                                        self.zbest[idx[i::skip]], 
                                        self.zp*1, get_err, 
                                        self.tempfilt, self.TEF,
                                        self.ARRAY_DTYPE, None)) 
                    for i in range(skip)]

        pool.close()
        pool.join()

        for res in jobs:
            _ = res.get(timeout=MULTIPROCESSING_TIMEOUT)
            _ix, _coeffs_best, _fmodel, _efmodel, _chi2_best, _cdraws = _
            self.coeffs_best[_ix,:] = _coeffs_best
            self.fmodel[_ix,:] = _fmodel
            self.efmodel[_ix,:] = _efmodel
            self.chi2_best[_ix] = _chi2_best
            if get_err:
                self.coeffs_draws[_ix,:,:] = _cdraws
    
    t1 = time.time()
    print(f'fit_best: {t1-t0:.1f} s (n_proc={np_check}, '
            f' NOBJ={subset.sum()})')


def _fit_at_zbest_group_single_template(ix, fnu_corr, efnu_corr, zbest, zp, get_err, tempfilt, TEF, ARRAY_DTYPE, _self):
    NOBJ = len(ix)
    
    NTEMP = tempfilt.NTEMP
    
    NDRAWS = 100
    if get_err > 1:
        NDRAWS = int(get_err)
    else:
        coeffs_draws = None
            
    if _self is None:
        coeffs_best = np.zeros((NOBJ, tempfilt.NTEMP), dtype=ARRAY_DTYPE)
        fmodel = np.zeros((NOBJ, tempfilt.NFILT), dtype=ARRAY_DTYPE)
        efmodel = np.zeros((NOBJ, tempfilt.NFILT), dtype=ARRAY_DTYPE)
        chi2_best = np.zeros(NOBJ, dtype=ARRAY_DTYPE)
        if get_err:
            coeffs_draws = np.zeros((NOBJ, NDRAWS, tempfilt.NTEMP),
                                dtype=ARRAY_DTYPE)
    else:
        # In place, avoid making copies
        coeffs_best = _self.coeffs_best
        fmodel = _self.fmodel
        efmodel = _self.efmodel
        chi2_best = _self.chi2_best
        if get_err:
            coeffs_draws = _self.coeffs_draws
        
    idx = np.where((zbest > tempfilt.zgrid[0]) & 
                   (zbest < tempfilt.zgrid[-1]))[0]
    
    for iobj in idx:
        
        zi = zbest[iobj]
        A = tempfilt(zi)
        TEFz = TEF(zi)

        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]
        if get_err:
            raise NotImplementedError('get_err not implemented')
        else:
            _ = fit_object_single_template(fnu_i, efnu_i, A, TEFz, zp, ARRAY_DTYPE)
            chi2, coeffs_best[iobj,:], fmodel[iobj,:] = _

        chi2_best[iobj] = chi2
    
    if _self is None:
        return ix, coeffs_best, fmodel, efmodel, chi2_best, coeffs_draws
    else:
        return True


def fit_catalog_single_template(
    self,
    idx=None,
    n_proc=4,
    verbose=False,
    get_best_fit=True,
    prior=False,
    beta_prior=False,
    **kwargs
):
    if "selection" in kwargs:
        idx = kwargs["selection"]

    if idx is None:
        idx_fit = self.idx
        selection = self.idx > -1
    else:
        if idx.dtype == bool:
            idx_fit = self.idx[idx]
            selection = idx
        else:
            idx_fit = idx
            selection = None

    # Setup
    fnu_corr = self.fnu[idx_fit, :] * self.ext_redden * self.zp
    efnu_corr = self.efnu[idx_fit, :] * self.ext_redden * self.zp

    missing = self.fnu[idx_fit, :] < self.param["NOT_OBS_THRESHOLD"]
    efnu_corr[missing] = self.param["NOT_OBS_THRESHOLD"] - 9.0

    t0 = time.time()
    if (n_proc == 0) | (mp.cpu_count() == 1):
        # Serial by redshift
        np_check = 1
        for iz, z in tqdm(enumerate(self.zgrid)):
            _res = fit_by_redshift_sigle_template(
                iz,
                self.zgrid[iz],
                self.tempfilt(self.zgrid[iz]),
                fnu_corr,
                efnu_corr,
                self.TEF(z),
                self.zp,
                self.param.params['VERBOSITY'],
                self.ARRAY_DTYPE,
            )

            self.chi2_fit[idx_fit, iz] = _res[1]
            self.fit_coeffs[idx_fit, iz, :] = _res[2]

    else:
        # With multiprocessing
        if n_proc < 0:
            np_check = mp.cpu_count()
        else:
            np_check = np.minimum(mp.cpu_count(), n_proc)

        pool = mp.Pool(processes=np_check)

        jobs = [
            pool.apply_async(
                fit_by_redshift_sigle_template,
                (
                    iz,
                    z,
                    self.tempfilt(z),
                    fnu_corr,
                    efnu_corr,
                    self.TEF(z),
                    self.zp,
                    self.param.params['VERBOSITY'],
                    self.ARRAY_DTYPE,
                ),
            )
            for iz, z in enumerate(self.zgrid)
        ]

        pool.close()

        # Gather results
        for res in tqdm(jobs):
            iz, chi2, coeffs = res.get(timeout=MULTIPROCESSING_TIMEOUT)
            self.chi2_fit[idx_fit, iz] = chi2
            self.fit_coeffs[idx_fit, iz, :] = coeffs

    # Compute maximum likelihood redshift zml
    if get_best_fit:
        if verbose:
            print("Compute best fits")

        fit_at_zbest_single_template(self, zbest=None, prior=prior, beta_prior=beta_prior)
    else:
        self.compute_lnp(prior=prior, beta_prior=beta_prior)

    t1 = time.time()
    if verbose:
        msg = "Fit {0:.1f} s (n_proc={1}, NOBJ={2})"
        print(msg.format(t1 - t0, np_check, len(idx_fit)))


def iterate_zp_templates_single_template(
    self,
    idx=None,
    update_templates=True,
    update_zeropoints=True,
    iter=0,
    n_proc=4,
    save_templates=False,
    error_residuals=False,
    prior=False,
    beta_prior=False,
    get_spatial_offset=False,
    spatial_offset_keys={"apply": True},
    verbose=False,
    **kwargs
):
    """
    Iterative detemination of zeropoint corrections
    """
    res = fit_catalog_single_template(
        self, idx=idx, n_proc=n_proc, prior=prior, beta_prior=beta_prior, verbose=verbose,
    )
    if error_residuals:
        self.error_residuals()

    if idx is not None:
        selection = np.zeros(self.NOBJ, dtype=bool)
        selection[idx] = True
    else:
        selection = None

    label = "{0}_zp_{1:03d}".format(self.param["MAIN_OUTPUT_FILE"], iter)

    fig = self.residuals(
        update_zeropoints=update_zeropoints,
        ref_filter=int(self.param["PRIOR_FILTER"]),
        selection=selection,
        update_templates=update_templates,
        full_label=label,
        **kwargs
    )

    fig_file = "{0}.png".format(label)
    fig.savefig(fig_file)

    if get_spatial_offset:
        self.spatial_statistics(
            catalog_mask=selection,
            output_suffix="_{0:03d}".format(iter),
            **spatial_offset_keys
        )

    if save_templates:
        self.save_templates()

def show_fit_single_template(self, id, id_is_idx=False, zshow=None, show_fnu=0, get_spec=False, xlim=[0.3, 9], show_components=False, show_redshift_draws=False, draws_cmap=None, ds9=None, ds9_sky=True, add_label=True, showpz=0.6, logpz=False, zr=None, axes=None, template_color='#1f77b4', figsize=[8,4], ndraws=100, fitter='nnls', show_missing=True, maglim=None, show_prior=False, show_stars=False, delta_chi2_stars=-20, max_stars=3, show_upperlimits=True, snr_thresh=2., with_tef=True, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.integrate import cumtrapz
    
    import astropy.units as u
    from cycler import cycler
    from eazy import igm as igm_module
    TEMPLATE_REDSHIFT_TYPE = 'nearest'
    
    IGM_OBJECT = igm_module.Inoue14()
    
    if id_is_idx:
        ix = id
    else:
        ix = self.idx[self.OBJID == id][0]
    
    if hasattr(self, 'h5file'):
        _data = self.get_object_data(ix)
        z, fnu_i, efnu_i, ra_i, dec_i, chi2_i, zspec_i, ok_i = _data
        lnp_i = -0.5*(chi2_i - np.nanmin(chi2_i))
        log_prior_i = np.ones(self.NZ)
        
    else:
        z = self.zbest[ix]
        fnu_i = self.fnu[ix, :]
        efnu_i = self.efnu[ix,:]
        ra_i = self.RA[ix]
        dec_i = self.DEC[ix]
        lnp_i = self.lnp[ix,:]
        log_prior_i = self.full_logprior[ix,:].flatten()
        chi2_i = self.chi2_fit[ix,:]
        zspec_i = self.ZSPEC[ix]
        ok_i = self.ok_data[ix,:]
        
    if zshow is not None:
        z = zshow
    
    if ds9 is not None:
        pan = ds9.get('pan fk5')
        if pan == '0 0':
            ds9_sky = False
            
        if ds9_sky:
            #for c in ['ra','RA','x_world']:
            pan = 'pan to {0} {1} fk5'
            ds9.set(pan.format(ra_i, dec_i))
        else:
            pan = 'pan to {0} {1}'
            ds9.set(pan.format(self.cat[self.fixed_cols['x']][ix], 
                                self.cat[self.fixed_cols['y']][ix]))
            
    ## SED        
    fnu_i = np.squeeze(fnu_i)*self.ext_redden*self.zp
    efnu_i = np.squeeze(efnu_i)*self.ext_redden*self.zp
    ok_band = (fnu_i/self.zp > self.param['NOT_OBS_THRESHOLD']) 
    ok_band &= (efnu_i/self.zp > 0)
    efnu_i[~ok_band] = self.param['NOT_OBS_THRESHOLD'] - 9.
    
    ## Evaluate coeffs at specified redshift
    tef_i = self.TEF(z)
    A = np.squeeze(self.tempfilt(z))
    chi2_i, coeffs_i, fmodel = fit_object_single_template(fnu_i, efnu_i, A, 
                                                tef_i, self.zp, 
                                                self.ARRAY_DTYPE)
    draws = None
    if draws is None:
        efmodel = 0
    else:
        efmodel = np.percentile(np.dot(draws, A), [16,84], axis=0)
        efmodel = np.squeeze(np.diff(efmodel, axis=0)/2.)
        
    ## Full SED
    templ = self.templates[0]
    tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]),
                        dtype=self.ARRAY_DTYPE)
    for i in range(self.NTEMP):
        zargs = {'z':z, 'redshift_type':TEMPLATE_REDSHIFT_TYPE}
        fnu = self.templates[i].flux_fnu(**zargs)*self.tempfilt.scale[i]
        try:
            tempflux[i, :] = fnu
        except:
            tempflux[i, :] = np.interp(templ.wave,
                                        self.templates[i].wave, fnu)
            
    templz = templ.wave*(1+z)

    if self.tempfilt.add_igm:
        igmz = templ.wave*0.+1
        lyman = templ.wave < 1300
        igmz[lyman] = IGM_OBJECT.full_IGM(z, templz[lyman])
    else:
        igmz = 1.

    templf = np.dot(coeffs_i, tempflux)*igmz
            
    if draws is not None:
        templf_draws = np.dot(draws, tempflux)*igmz
            
    fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
    
    if show_fnu:
        if show_fnu == 2:
            templz_power = -1
            flam_spec = 1.e29/(templz/1.e4)
            flam_sed = 1.e29/self.ext_corr/(self.pivot/1.e4)
            ylabel = (r'$f_\nu / \lambda$ [$\mu$Jy / $\mu$m]')
            flux_unit = u.uJy / u.micron
        else:
            templz_power = 0
            flam_spec = 1.e29
            flam_sed = 1.e29/self.ext_corr
            ylabel = (r'$f_\nu$ [$\mu$Jy]')    
            flux_unit = u.uJy
        
    else:
        templz_power = -2
        flam_spec = utils.CLIGHT*1.e10/templz**2/1.e-19
        flam_sed = utils.CLIGHT*1.e10/self.pivot**2/self.ext_corr/1.e-19
        ylabel = (r'$f_\lambda [10^{-19}$ erg/s/cm$^2$]')
        
        flux_unit = 1.e-19*u.erg/u.s/u.cm**2/u.AA
                    
    try:
        from collections import OrderedDict
        data = OrderedDict(ix=ix, id=self.OBJID[ix], z=z,
                        z_spec=zspec_i, 
                        pivot=self.pivot, 
                        model=fmodel*fnu_factor*flam_sed,
                        emodel=efmodel*fnu_factor*flam_sed,
                        fobs=fnu_i*fnu_factor*flam_sed, 
                        efobs=efnu_i*fnu_factor*flam_sed,
                        valid=ok_i,
                        tef=tef_i,
                        templz=templz,
                        templf=templf*fnu_factor*flam_spec,
                        show_fnu=show_fnu*1,
                        flux_unit=flux_unit,
                        wave_unit=u.AA, 
                        chi2=chi2_i, 
                        coeffs=coeffs_i)
    except:
        data = None
    
    ## Just return the data    
    if get_spec:            
        return data
    
    ###### Make the plot
    
    if axes is None:
        fig = plt.figure(figsize=figsize)
        if showpz:
            fig_axes = GridSpec(1,2,width_ratios=[1,showpz])
        else:    
            fig_axes = GridSpec(1,1,width_ratios=[1])
            
        ax = fig.add_subplot(fig_axes[0])
    else:
        fig = None
        fig_axes = None
        ax = axes[0]
                    
    ax.scatter(self.pivot/1.e4, fmodel*fnu_factor*flam_sed, 
                color='w', label=None, zorder=1, s=120, marker='o')
    
    ax.scatter(self.pivot/1.e4, fmodel*fnu_factor*flam_sed, marker='o',
                color=template_color, label=None, zorder=2, s=50, 
                alpha=0.8)

    if draws is not None:
        ax.errorbar(self.pivot/1.e4, fmodel*fnu_factor*flam_sed,
                    efmodel*fnu_factor*flam_sed, alpha=0.8,
                    color=template_color, zorder=2,
                    marker='None', linestyle='None', label=None)
    
    # Missing data
    missing = (fnu_i < self.param['NOT_OBS_THRESHOLD']) 
    missing |= (efnu_i < 0)
    
    # Detection
    sn2_detection = (~missing) & (fnu_i/efnu_i > snr_thresh)
    
    # S/N < 2
    sn2_not = (~missing) & (fnu_i/efnu_i <= snr_thresh)
    
    # Uncertainty with TEF
    if with_tef:
        err_tef = np.sqrt(efnu_i**2+(tef_i*fnu_i)**2)            
    else:
        err_tef = efnu_i*1
        
    ax.errorbar(self.pivot[sn2_detection]/1.e4, 
                (fnu_i*fnu_factor*flam_sed)[sn2_detection], 
                (err_tef*fnu_factor*flam_sed)[sn2_detection], 
                color='k', marker='s', linestyle='None', label=None, 
                zorder=10)

    if show_upperlimits:
        ax.errorbar(self.pivot[sn2_not]/1.e4, 
                    (fnu_i*fnu_factor*flam_sed)[sn2_not], 
                    (efnu_i*fnu_factor*flam_sed)[sn2_not], color='k', 
                    marker='s', alpha=0.4, linestyle='None', label=None)

    pl = ax.plot(templz/1.e4, templf*fnu_factor*flam_spec, alpha=0.5, 
                    zorder=-1, color=template_color, 
                    label='z={0:.2f}'.format(z))
    
    if show_components:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i in range(self.NTEMP):
            if coeffs_i[i] != 0:
                pi = ax.plot(templz/1.e4, 
                    coeffs_i[i]*tempflux[i,:]*igmz*fnu_factor*flam_spec, 
                            alpha=0.5, zorder=-1, 
                            label=self.templates[i].name.split('.dat')[0], 
                            color=colors[i % len(colors)])
                        
    elif show_redshift_draws:
        
        if draws_cmap is None:
            draws_cmap = plt.cm.rainbow
            
        # Draw random values from p(z)
        pz = np.exp(lnp_i).flatten()
        pzcum = cumtrapz(pz, x=self.zgrid)
        
        if show_redshift_draws == 1:
            nzdraw = 100
        else:
            nzdraw = show_redshift_draws*1
        
        rvs = np.random.rand(nzdraw)
        zdraws = np.interp(rvs, pzcum, self.zgrid[1:])
        
        for zi in zdraws:
            Az = np.squeeze(self.tempfilt(zi))
            chi2_zi, coeffs_zi, fmodelz = fit_object_single_template(fnu_i, efnu_i, 
                                                    Az, 
                                                    self.TEF(zi), self.zp, 
                                                    self.ARRAY_DTYPE)
                                                    
            c_i = np.interp(zi, self.zgrid, np.arange(self.NZ)/self.NZ)
            
            templzi = templ.wave*(1+zi)
            if self.tempfilt.add_igm:
                igmz = templ.wave*0.+1
                lyman = templ.wave < 1300
                igmz[lyman] = IGM_OBJECT.full_IGM(zi, templzi[lyman])
            else:
                igmz = 1.

            templfz = np.dot(coeffs_zi, tempflux)*igmz                
            templfz *=  flam_spec * (templz / templzi)**templz_power
            
            plz = ax.plot(templzi/1.e4, templfz*fnu_factor,
                            alpha=np.maximum(0.1, 1./nzdraw), 
                            zorder=-1, color=draws_cmap(c_i))
            
    if draws is not None:
        templf_width = np.percentile(templf_draws*fnu_factor*flam_spec, 
                                        [16,84], axis=0)
        ax.fill_between(templz/1.e4, templf_width[0,:], templf_width[1,:], 
                        color=pl[0].get_color(), alpha=0.1, label=None)
                                                    
    if show_stars & (not hasattr(self, 'star_chi2')):
        print('`star_chi2` attribute not found, run `fit_phoenix_stars`.')
        
    elif show_stars & hasattr(self, 'star_chi2'):
        # if __name__ == '__main__':
        #     # debug
        #     ix = _[1]['ix']
        #     chi2_i = self.chi2_noprior[ix]  
        #     ax = _[0].axes[0]
            
        delta_chi2 = self.star_chi2[ix,:] - chi2_i    
        good_stars = delta_chi2 < delta_chi2_stars
        good_stars &= (self.star_chi2[ix,:] - self.star_chi2[ix,:].min() < 100)
        
        if good_stars.sum() == 0:
            msg = 'Min delta_chi2 = {0:.1f} ({1})'
            sname = self.star_templates[np.argmin(delta_chi2)].name
            print(msg.format(delta_chi2.min(), sname))
            
        else:
            # dummy for cycler
            ax.plot(np.inf, np.inf)
            star_models  = self.star_tnorm[ix,:] * self.star_flux
            so = np.argsort(self.pivot)
            order = np.where(good_stars)[0]
            order = order[np.argsort(delta_chi2[order])]
        
            for si in order[:max_stars]:
                label = self.star_templates[si].name.strip('bt-settl_')
                label = '{0} {1:5.1f}'.format(label.replace('_', ' '),
                                                delta_chi2[si])
                print(label)
                ax.plot(self.pivot[so]/1.e4,
                        (star_models[:,si]*fnu_factor*flam_sed)[so], 
                        marker='o', alpha=0.5, label=label)

            if __name__ == '__main__':
                ax.legend()
            
    if axes is None:            
        ax.set_ylabel(ylabel)
        
        if sn2_detection.sum() > 0:
            ymax = (fmodel*fnu_factor*flam_sed)[sn2_detection].max()
        else:
            ymax = (fmodel*fnu_factor*flam_sed).max()
                    
        if np.isfinite(ymax):
            ax.set_ylim(-0.1*ymax, 1.2*ymax)

        ax.set_xlim(xlim)
        xt = np.array([0.1, 0.5, 1, 2, 4, 8, 24, 160, 500])*1.e4

        ax.semilogx()

        valid_ticks = (xt > xlim[0]*1.e4) & (xt < xlim[1]*1.e4)
        if valid_ticks.sum() > 0:
            xt = xt[valid_ticks]
            ax.set_xticks(xt/1.e4)
            ax.set_xticklabels(xt/1.e4)

        ax.set_xlabel(r'$\lambda_\mathrm{obs}$')
        ax.grid()
        
        if add_label:
            txt = '{0}\nID={1}'
            txt = txt.format(self.param['MAIN_OUTPUT_FILE'], 
                                self.OBJID[ix]) #, self.prior_mag_cat[ix])
                                
            ax.text(0.95, 0.95, txt, ha='right', va='top', fontsize=7,
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='w', alpha=0.5), zorder=10)
            
            ax.legend(fontsize=7, loc='upper left')
    
    # Optional mag scaling if show_fnu = 1 for uJy
    if (maglim is not None) & (show_fnu == 1):
        
        ax.semilogy()
        # Limits
        ax.scatter(self.pivot[sn2_not]/1.e4,
                    ((3*efnu_i)*fnu_factor*flam_sed)[sn2_not], 
                    color='k', marker='v', alpha=0.4, label=None)
        
        # Mag axes
        axm = ax.twinx()
        ax.set_ylim(10**(-0.4*(np.array(maglim)-23.9)))
        axm.set_ylim(0,1)
        ytv = np.arange(maglim[0], maglim[1], -1, dtype=int)
        axm.set_yticks(np.interp(ytv, maglim[::-1], [1,0]))
        axm.set_yticklabels(ytv)
    
    if show_missing:
        yl = ax.get_ylim()
        ax.scatter(self.pivot[missing]/1.e4,
                    self.pivot[missing]*0.+yl[0],
                    marker='h', s=120,
                    fc='None', ec='0.7',
                    alpha=0.6,
                    zorder=-100)
    
    ## P(z)
    if not showpz:
        return fig, data
        
    if axes is not None:
        if len(axes) == 1:
            return fig, data
        else:
            ax = axes[1]
    else:
        ax = fig.add_subplot(fig_axes[1])
    
    chi2 = np.squeeze(chi2_i)
    prior = np.exp(log_prior_i)
    #pz = np.exp(-(chi2-chi2.min())/2.)*prior
    #pz /= np.trapz(pz, self.zgrid)
    pz = np.exp(lnp_i).flatten()
    
    ax.plot(self.zgrid, pz, color='orange', label=None)
    if show_prior:
        ax.plot(self.zgrid, prior/prior.max()*pz.max(), color='g',
            label='prior')
    
    ax.fill_between(self.zgrid, pz, pz*0, color='yellow', alpha=0.5, 
                    label=None)
    if zspec_i > 0:
        ax.vlines(zspec_i, 1.e-5, pz.max()*1.05, color='r',
                    label='zsp={0:.3f}'.format(zspec_i))
    
    if zshow is not None:
        ax.vlines(zshow, 1.e-5, pz.max()*1.05, color='purple', 
                    label='z={0:.3f}'.format(zshow))
        
    if axes is None:
        ax.set_ylim(0,pz.max()*1.05)
        
        if logpz:
            ax.semilogy()
            ymax = np.minimum(ax.get_ylim()[1], 100)
            ax.set_ylim(1.e-3*ymax, 1.8*ymax)
            
        if zr is None:
            ax.set_xlim(0,self.zgrid[-1])
        else:
            ax.set_xlim(zr)
            
        ax.set_xlabel('z'); ax.set_ylabel('p(z)')
        ax.grid()
        ax.set_yticklabels([])
        
        fig_axes.tight_layout(fig, pad=0.5)
        
        # if add_label & (zspec_i > 0):
        ax.legend(fontsize=7, loc='upper left')
            
        return fig, data
    else:
        return fig, data
    

def pz_percentiles(self, percentiles=[2.5, 16, 50, 84, 97.5], oversample=5, selection=None,
                   return_pit_crps=False):
    from eazy.utils import log_zgrid
    import scipy.interpolate 
    try:
        from scipy.integrate import cumtrapz
    except ImportError:
        from scipy.integrate import cumulative_trapezoid as cumtrapz

    interpolator = scipy.interpolate.Akima1DInterpolator

    p100 = np.array(percentiles)/100.
    zlimits = np.zeros((self.NOBJ, p100.size), dtype=self.ARRAY_DTYPE)
        
    zr = [self.param['Z_MIN'], self.param['Z_MAX']]
    zgrid_zoom = log_zgrid(zr=zr,dz=self.param['Z_STEP']/oversample)
            
    ok = self.zbest > self.zgrid[0]      
    if selection is not None:
        ok &= selection

    if ok.sum() == 0:
        print('pz_percentiles: No objects in selection')
        
    spl = interpolator(self.zgrid, self.lnp[ok,:], axis=1)

    pz_zoom = np.exp(spl(zgrid_zoom))

    # Akima1DInterpolator can get some NaNs at the end?
    valid = np.isfinite(pz_zoom)
    pz_zoom[~valid] = 0.

    pzcum = cumtrapz(pz_zoom, x=zgrid_zoom, axis=1)

    pzcmax = pzcum.max(axis=1)
    pzcum = (pzcum.T / pzcmax).T

    for j, i in enumerate(self.idx[ok]):
        zlimits[i,:] = np.interp(p100, pzcum[j, :], zgrid_zoom[1:])
    
    if return_pit_crps:
        pit = np.zeros((self.NOBJ), dtype=self.ARRAY_DTYPE)
        crps = np.zeros((self.NOBJ), dtype=self.ARRAY_DTYPE)
        for j, i in enumerate(self.idx[ok]):
            pit[i] = np.interp(self.ZSPEC[i], zgrid_zoom[1:], pzcum[j,:])
            hstepf = np.zeros_like(pzcum[j,:])
            hstepf[zgrid_zoom[1:] > self.ZSPEC[i]] = 1
            crps[i] = np.trapz((pzcum[j,:]-hstepf)**2, x=zgrid_zoom[1:])

        return zlimits, pit, crps
    else:
        return zlimits
    
    
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
    figsave=True,
    ax=None,
    rax=None,
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
    # print(f"Objects : {np.sum(z_cnd):d}")

    delta_z = z_phot - z_spec
    dz = delta_z / (1 + z_spec)
    bias = np.mean(dz[z_cnd])
    # Normalized Median Absolute Deviation (NMAD)
    nmad = 1.48 * np.median(
        np.abs(delta_z[z_cnd] - np.median(delta_z[z_cnd])) / (1 + z_spec[z_cnd])
    )
    sigma = np.std(dz[z_cnd])

    outlier = z_cnd & (np.abs(dz) >= 0.15)
    # print(f"Outliers: {np.sum(outlier):d}")
    # print("\n")

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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        if (ax is None) and (rax is None):
            fig, axes = plt.subplots(2, 1, figsize=figsize,
                                gridspec_kw={"height_ratios": [3, 1], "hspace":0})
            ax = axes[0]
            rax = axes[1]
        elif (ax is not None) and (rax is not None):
            pass
        else:
            raise ValueError("Both ax and rax must be provided if residual_plot is True.")
            
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
                yerr = sigz[z_cnd]*(1+z_phot[z_cnd])
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
                                 ec="k", cmap="Blues_r", vmin=0, vmax=3, zorder=2)
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
    if figsave:
        plt.savefig(out, dpi=300)
    if figclose:
        plt.close()

    return ids[outlier]


def get_result_figures(base, figdir, scheme, pit=False, pitmask=None, pitmaskdesc=""):
    id_out = plot_comp_hexbin(base['z_true'], base['z_phot'], base['z_phot_chi2'],
                            figdir/'all_hexhist.png', base['id'],
                            z_160=base['z160'], z_840=base['z840'],
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
    
    if pit:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(base["pit"], bins=50, histtype="stepfilled", lw=1, color="cornflowerblue")
        ax.set_xlabel("PIT")
        ax.set_ylabel(r"$N$")
        ax.set_title(f"PIT ({scheme})")
        fig.tight_layout()
        fig.savefig(figdir/"pit_hist.png")
        plt.close()
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(np.log10(base["crps"]), bins=50, histtype="stepfilled", lw=1, color="cornflowerblue")
        ax.set_xlabel(r"$\log$ CRPS")
        ax.set_ylabel(r"$N$")
        ax.set_title(f"CRPS ({scheme})")
        ax.text(0.95, 0.95, f"Mean: {np.mean(base['crps']):.2e}", transform=ax.transAxes, ha="right", va="top")
        fig.tight_layout()
        fig.savefig(figdir/"crps_hist.png")
        plt.close()
        
        if pitmask is not None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.hist(base["pit"][pitmask], bins=30, histtype="stepfilled", lw=1, color="cornflowerblue")
            ax.set_xlabel("PIT")
            ax.set_ylabel(r"$N$")
            ax.set_title(f"PIT [{pitmaskdesc}] ({scheme})")
            fig.tight_layout()
            fig.savefig(figdir/"pit_hist_masked.png")
            plt.close()
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.hist(np.log10(base["crps"][pitmask]), bins=30, histtype="stepfilled", lw=1, color="cornflowerblue")
            ax.set_xlabel(r"$\log$ CRPS")
            ax.set_ylabel(r"$N$")
            ax.set_title(f"CRPS [{pitmaskdesc}] ({scheme})")
            ax.text(0.95, 0.95, f"Mean: {np.mean(base['crps'][pitmask]):.2e}", transform=ax.transAxes, ha="right", va="top")
            fig.tight_layout()
            fig.savefig(figdir/"crps_hist_masked.png")