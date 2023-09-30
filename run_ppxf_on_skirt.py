"""
This python file runs ppxf on the output of skirt runs.

REQUIRES: numpy, scipy, matplotlib, astropy, h5py, ppxf, vorbin, plotbin, fitsdatacube (https://github.com/JaedenBardati/skirt-datacube)
Optionally requires tqdm for a progress bar, and/or my_timing (https://gist.github.com/JaedenBardati/e953033508000f637a4121982429a56e) for logging.

Jaeden Bardati 2023, adapted from some code by John Ruan.

Example usage: python "run_ppxf_on_skirt.py -i 6400_39_2 ... -o ppxf_output_6400_39_2.h5 ..."
"""

# built in modules
from os import path

# standard pip modules
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import h5py

# ppxf, vorbin (pip modules by Michele Cappellari)
from ppxf.ppxf import ppxf
import ppxf.miles_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# my custom packages
from fitsdatacube import FitsDatacube, Filter, GaussianPSF     # for loading and manipulating skirt data, available at https://github.com/JaedenBardati/skirt-datacube



def _run_ppxf(j, bin_num, good_spaxels_spectra, good_spaxels_spectra_errorbar, stars_templates, velscale, 
                start, moments, ppxf_wav, lam_temp, component, dust, regul, sigma_diff2):
    """Run ppxf in a separate global function for parallel running."""
    w = bin_num == j
    
    # get the sum of the spectra in the current voronoi bin to fit
    galaxy = np.sum(good_spaxels_spectra[:, w], 1)
    galaxy_noise = np.sqrt(np.sum(good_spaxels_spectra_errorbar[:, w]*good_spaxels_spectra_errorbar[:, w], 1))
    
    # pick out the good wavelength bins in the summed spectrum to fit
    goodpixels = np.where(~np.isnan(galaxy))[0]
    
    # fit the summed spectrum with ppxf
    pp = ppxf(stars_templates, galaxy, galaxy_noise, velscale, start, goodpixels=goodpixels,
            moments=moments, degree=12, mdegree=2, lam=ppxf_wav, lam_temp=lam_temp,
            component=component, dust=dust, regul=regul, reg_ord=1, clean=False, quiet=1)
    
    sigma_obs = pp.sol[1]   # sigma is second element of first kinematic component
    sigma = np.sqrt(sigma_obs**2 - sigma_diff2)
    errors = pp.error*np.sqrt(pp.chi2)      # assume the fit is good

    return j, pp.sol[0], sigma, errors[0], errors[1]  # vel, sig, vel err, sig err


def run_ppxf_individual(root_file, hf_file, nphotons, psf=None, filters=None, sim_ext='total', ppxf_wavmin=3450, ppxf_wavmax=6000, 
                        pixsize=0.1, r_vov_cutoff=0.1, frac_reliable=0.9, target_sn=50, iunits=('Jy', 'micron'), 
                        parallel=True, progress_bar=False, log=False, plot=False):
    """
    Loads the SKIRT output files (for a given instrument), runs ppxf (using the E-MILES stellar templates) on it, voronoi bins
    the spectrum to a target minimum S/N and outputs the resulting stellar LOS velocity and sigma maps in an hdf5 file.
    Note, it also requires the stats (1-4) files to determine the R and VOV to mask the error in the spectrum (outputted by setting recordStatistics=true).
    Assumes square images.

    Arguments:

     [root_file]    :   Extension-less filename for the main skirt output file (e.g. 'MonoDisk_i88' for the file 'MonoDisk_i88_total.fits').
     [hf_file]      :   Output hdf5 filename.
     [nphotons]     :   Number of photons used as a parameter in the skirt simulation, used for R and VOV calulation.

     (psf)          :   PSF to convolve the datacube with prior to all operations (including ppxf and image convolution). 
     (filters)      :   A list of filter objects to produce band-convolved images which are stored in the hdf5 file (see examples below). 
                        Leave as None to avoid band-convolving.
     (sim_ext)      :   Simulation output extension for the main skirt output file (e.g. 'total' for the file 'MonoDisk_i88_total.fits'). 
                        Change if you want something like 'transparent' (outputted by setting recordComponents=true).
     (ppxf_wavmin)  :   Minimum wavelength to run ppxf fitting on (in angstroms). It is best to restrict ppxf to a region of the spectrum with a lot 
                        of emission lines to reduce computation time and fitting the continuum. Defaults to a good optical range.
     (ppxf_wavmin)  :   Maximum wavelength to run ppxf fitting on (in angstroms). Ditto.
     (pixsize)      :   Number of arcseconds corresponding to one pixel. 
     (r_vov_cutoff) :   Cutoff of R and VOV below which pixels are determined as unreliable (see https://skirt.ugent.be/root/_user_statistics.html for details).
     (frac_reliable):   Fraction of reliable pixels required in a spaxel before it is masked. 
     (target_sn)    :   Target S/N for the voronoi bins.
     (iunits)       :   Input unit system of the skirt output files in the form: (flux density or surface brightness, wavelength).
                        The unit system is handled by astropy, so you can enter custom astropy units instead of strings if needed.
     (parallel)     :   Flag to run ppxf in parallel using multiprocessing.
     (progress_bar) :   Flag to display a progress bar during the ppxf fitting. 
     (log)          :   Flag to print a log including the timing. 
     (plot)         :   Flag to plots some figures. 
    
    """
    if filters is not None:
        for _filter in filters:
            if Filter not in type(_filter).__mro__:
                raise TypeError('The filters must be of the type %s.' % str(Filter))
    
    ##############################################################################
    if log: 
        from my_timing import log_timing
        log_timing('Loading SKIRT data from {}...'.format(root_file))

    FILENAME = root_file + '_' + sim_ext + '.fits'     # main file
    FILENAME_STATS1 = root_file + '_stats1.fits'        # stats files
    FILENAME_STATS2 = root_file + '_stats2.fits'
    FILENAME_STATS3 = root_file + '_stats3.fits'
    FILENAME_STATS4 = root_file + '_stats4.fits'
    UNITS = ('Jy', 'angstrom')
    C = 299792.458  # speed of light in km/s
    
    # load fits datacubes
    fdc = FitsDatacube(FILENAME, iunits=iunits, ounits=UNITS)  # note that some units may need to be converted manually if they assume something about the data 
    fdc_stats1 = FitsDatacube(FILENAME_STATS1, ounits=UNITS)  # input units probably shouldn't matter for stats files, since we are only interested in their relative measurements
    fdc_stats2 = FitsDatacube(FILENAME_STATS2, ounits=UNITS)
    fdc_stats3 = FitsDatacube(FILENAME_STATS3, ounits=UNITS)
    fdc_stats4 = FitsDatacube(FILENAME_STATS4, ounits=UNITS)
    assert fdc.shape == fdc_stats1.shape == fdc_stats2.shape == fdc_stats3.shape == fdc_stats4.shape, 'One or more of the files have mismatching data shapes.'

    if psf is not None:
        # convolve with psf
        fdc = fdc.convolve_PSF(psf)

    # compute error bars on spectrum
    R = (fdc_stats2/fdc_stats1**2 - 1/nphotons)**0.5
    VOV = (fdc_stats4 - 4*fdc_stats1*fdc_stats3/nphotons + 8*fdc_stats2*fdc_stats1**2/nphotons**2 - 4*fdc_stats1**4/nphotons**3 - fdc_stats2**2/nphotons)/(fdc_stats2 - fdc_stats1**2/nphotons)**2

    # clip wavelength range to optical
    ppxf_fdc = fdc[np.where((fdc.wav>=ppxf_wavmin) & (fdc.wav<=ppxf_wavmax)),:,:][0]
    ppxf_R = R[np.where((fdc.wav>=ppxf_wavmin) & (fdc.wav<=ppxf_wavmax)),:,:][0]
    ppxf_VOV = VOV[np.where((fdc.wav>=ppxf_wavmin) & (fdc.wav<=ppxf_wavmax)),:,:][0]
    ppxf_errorbar = ppxf_fdc*R[np.where((fdc.wav>=ppxf_wavmin) & (fdc.wav<=ppxf_wavmax)),:,:][0]
    ppxf_wav = fdc.wav[np.where((fdc.wav>=ppxf_wavmin) & (fdc.wav<=ppxf_wavmax))]

    # Transform cube into 2-dim array of spectra so I can loop through them to fit them individually 
    npix = ppxf_fdc.shape[0] #number of wavelength bins in each spectrum
    sidelength = ppxf_fdc.shape[1] # number of pixels on each side 
    nspectra = sidelength*sidelength #total number of spectra in fdc
    spectra = ppxf_fdc.reshape(npix, -1) # create array of spectra [npix, nx*ny]
    spectra = C*1e-10*spectra/(ppxf_wav**2)[:, np.newaxis] # convert from Jansky to erg/cm^2/s/Ang
    spectra_R = ppxf_R.reshape(npix, -1)
    spectra_VOV = ppxf_VOV.reshape(npix, -1)
    spectra_errorbar = ppxf_errorbar.reshape(npix, -1) # create array of spectra [npix, nx*ny]
    spectra_errorbar = C*1e-10*spectra_errorbar/(ppxf_wav**2)[:, np.newaxis]# convert from Jansky to erg/cm^2/s/Ang

    ##############################################################################
    if log: log_timing('Masking out bad spaxels and creating voronoi bins...')

    # mask out bad the spaxels spectra first before voronoi binning by replacing all their flux density values with NaN
    for i in range(0, nspectra):
        galaxy_R = spectra_R[:,i]
        galaxy_VOV = spectra_VOV[:,i]
        if len(galaxy_VOV[np.where(galaxy_VOV<=r_vov_cutoff) and np.where(galaxy_R<=r_vov_cutoff)])/float(len(galaxy_VOV)) < frac_reliable:    
            for j in range(0, len(ppxf_wav)):
                spectra[j,i] = np.nan

    # Create coordinates centred on the brightest spectrum
    flux = np.nanmean(spectra, axis=0)
    jm = np.argmax(flux)
    row, col = map(np.ravel, np.indices(ppxf_fdc.shape[-2:]))
    x = (col - col[jm])*pixsize
    y = (row - row[jm])*pixsize

    # calculate the 2D signal and noise map in the datacube for the wavelength-integrated image (used as a proxy S/N map for voronoi binning)
    spaxel_signal = flux
    spaxel_noise = np.sqrt(np.nanmean(spectra_errorbar**2, axis=0))

    # pick out all the reliable pixels from this 2D map
    good_bin = np.where(~np.isnan(spaxel_signal))
    good_spaxel_signal = spaxel_signal[good_bin]
    good_spaxel_noise = spaxel_noise[good_bin]
    x = x[good_bin]
    y = y[good_bin]

    # use this signal and noise map to do voronoi tessellations
    if plot: plt.figure(figsize=(7,10))
    bin_num, x_gen, y_gen, xbin, ybin, sn, nPixels, scale = voronoi_2d_binning(x, y, good_spaxel_signal, good_spaxel_noise, target_sn, plot=int(plot), quiet=1)

    # pick out all the reliable spaxels from spectra datacube
    good_spaxels = np.where(~np.isnan(spectra))
    good_spaxels_spectra = np.reshape(spectra[good_spaxels], (len(ppxf_wav), len(good_spaxel_signal)))
    good_spaxels_spectra_errorbar = np.reshape(spectra_errorbar[good_spaxels],  (len(ppxf_wav), len(good_spaxel_signal)))

    ##############################################################################
    if log: log_timing('Setting up stellar templates...')

    spec_res = fdc.infer_R()  # inferred spectral resolution 
    FWHM_gal = np.sqrt(min(ppxf_wav)*max(ppxf_wav))/spec_res
    FWHM_temp = 2.51 # resolution of E-MILES templates in the fitted range
    velscale = C*np.log(ppxf_wav[1]/ppxf_wav[0]) # eq.(8) of Cappellari (2017)

    ppxf_dir = path.dirname(path.realpath(lib.__file__))
    pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
    miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950], age_range=[0, 10.])

    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    start = [0, 150.]  # (km/s), starting guess for [V,sigma]
    moments = 2  # we only want to get V and sigma

    # set regularization
    Delta = 0.01
    regul = 1./Delta

    #normalize templates
    for j in range(0,len(stars_templates[0,:])):
        stars_templates[:,j] /= np.mean(stars_templates[:,j][np.where((miles.lam_temp>5070) & (miles.lam_temp<5950))])

    n_stars = stars_templates.shape[1]
    component = [0]*n_stars 

    # set dust parameters
    # assumes the default attenuation curve in Cappellari (2022), but we only use the first two parameters (A_V, delta)
    stellar_component = np.array(component) < 1    # assuming stars = 0 and gas = 1
    dust_stars = {"start": [0.1, -0.1], "bounds": [[0, 4], [-1, 0.4]], "component": stellar_component}
    dust = [dust_stars]

    # pre-calculate some numbers for ppxf
    lam_med = np.median(ppxf_wav)  # in angstroms
    sigma_gal = C*FWHM_gal/lam_med/2.355  # in km/s
    sigma_temp = C*FWHM_temp/lam_med/2.355
    sigma_diff2 = sigma_gal**2 - sigma_temp**2   # eq. (5) of Cappellari (2017)

    ##############################################################################
    if log: log_timing('Fitting stellar templates using ppxf...')
    
    nbins = sn.size
    velbin, sigbin, velbin_err, sigbin_err = np.zeros((4, nbins))
    rangebins = range(nbins)

    if not parallel:  # if serial
        if progress_bar: 
            from tqdm.notebook import tqdm
            rangebins = tqdm(rangebins)

        for j in rangebins:
            _j, velbin[j], sigbin[j], velbin_err[j], sigbin_err[j] = _run_ppxf(j, bin_num, good_spaxels_spectra, good_spaxels_spectra_errorbar, 
                                                                               stars_templates, velscale, start, moments, ppxf_wav, miles.lam_temp, 
                                                                               component, dust, regul, sigma_diff2)
            assert _j == j, 'Something very weird has happened...'
    else:  # if parallel
        from multiprocessing import Pool
        from itertools import repeat

        args = zip(rangebins, repeat(bin_num), repeat(good_spaxels_spectra), repeat(good_spaxels_spectra_errorbar), 
                   repeat(stars_templates), repeat(velscale), repeat(start), repeat(moments), repeat(ppxf_wav), 
                   repeat(miles.lam_temp), repeat(component), repeat(dust), repeat(regul), repeat(sigma_diff2))
        if progress_bar: 
            from tqdm.notebook import tqdm
            args = tqdm(args, total=nbins)

        with Pool() as pool:
            for j, _vel, _sig, _vel_err, _sig_err in pool.starmap(_run_ppxf, args, chunksize=1):
                velbin[j], sigbin[j], velbin_err[j], sigbin_err[j] = _vel, _sig, _vel_err, _sig_err
        
        del args  # closes tqdm if needed
    del rangebins  # closes tqdm if needed

    ##############################################################################
    if log: log_timing('Reformatting ppxf output...')

    # output these binned kinematic maps as a masked 2D array that can be plotted using imshow
    # the final_vel_masked and final_sigma_masked 2D arrays are what you want

    velocity_array = np.full(shape=(sidelength, sidelength), fill_value=np.nan)
    sigma_array = np.full(shape=(sidelength, sidelength), fill_value=np.nan)
    velocity_err_array = np.full(shape=(sidelength, sidelength), fill_value=np.nan)
    sigma_err_array = np.full(shape=(sidelength, sidelength), fill_value=np.nan)

    # map the coordinates of the pixels back to the original fdc indicies
    for i in range(0, len(x)): #for each pixel, map it on to a pixel in velocity_array
        new_col = int(np.round(x[i]/pixsize + col[jm]))
        new_row = int(np.round(y[i]/pixsize + row[jm]))
        vel = velbin[bin_num[i]]
        sig = sigbin[bin_num[i]]
        vel_err = velbin_err[bin_num[i]]
        sig_err = sigbin_err[bin_num[i]] 
        velocity_array[new_row, new_col] = vel
        sigma_array[new_row, new_col] = sig
        velocity_err_array[new_row, new_col] = vel_err
        sigma_err_array[new_row, new_col] = sig_err

    # mask out the poor-S/N pixels that were not binned
    final_vel_masked = np.ma.masked_where(np.isnan(velocity_array), velocity_array)
    final_sigma_masked = np.ma.masked_where(np.isnan(sigma_array), sigma_array)
    final_vel_err = np.ma.masked_where(np.isnan(velocity_err_array), velocity_err_array)
    final_sigma_err = np.ma.masked_where(np.isnan(sigma_err_array), sigma_err_array)
    final_mask = np.isnan(velocity_array) | np.isnan(sigma_array) | np.isnan(velocity_err_array) | np.isnan(sigma_err_array)

    # The resulting sigma and velocity maps may have disconnected regions
    # Apply an additional mask for these disconnected regions, keeping only the biggest region
    labels, num_labels = scipy.ndimage.label(~final_vel_masked.mask)
    largest_component_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_component_mask = (labels == largest_component_label)
    final_continous_mask = final_vel_masked.mask | ~largest_component_mask
    final_vel_masked = np.ma.array(data=final_vel_masked.data, mask=final_continous_mask)
    final_sigma_masked = np.ma.array(data=final_sigma_masked.data, mask=final_continous_mask)
    final_vel_err = np.ma.array(data=final_vel_err.data, mask=final_continous_mask)
    final_sigma_err = np.ma.array(data=final_sigma_err.data, mask=final_continous_mask)

    extent = (-pixsize*sidelength/2., pixsize*sidelength/2., -pixsize*sidelength/2., pixsize*sidelength/2.)

    if plot:
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle('LOSVD', fontsize=15)

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = plt.imshow(final_vel_masked, cmap='RdBu_r', extent=extent, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$v_\mathrm{los}$ [km s$^{-1}$]', fontsize=13) 
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_xlabel(r'$\Delta$X [arcsec]', size=12)
        ax1.set_ylabel(r'$\Delta$Y [arcsec]', size=12)

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = plt.imshow(final_sigma_masked, cmap='RdBu_r', extent=extent, origin='lower')
        cbar2 = fig.colorbar(im2, ax=ax2) 
        cbar2.set_label(r'$\sigma_\star$ [km s$^{-1}$]', fontsize=13) 
        cbar2.ax.tick_params(labelsize=12)
        ax2.set_xlabel(r'$\Delta$X [arcsec]', size=12)
        ax2.set_ylabel(r'$\Delta$Y [arcsec]', size=12)

        fig = plt.figure(figsize=(10, 3))
        fig.suptitle('LOSVD error', fontsize=15)

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = plt.imshow(final_vel_err, cmap='RdBu_r', extent=extent, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$v_\mathrm{los}$ [km s$^{-1}$]', fontsize=13) 
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_xlabel(r'$\Delta$X [arcsec]', size=12)
        ax1.set_ylabel(r'$\Delta$Y [arcsec]', size=12)

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = plt.imshow(final_sigma_err, cmap='RdBu_r', extent=extent, origin='lower')
        cbar2 = fig.colorbar(im2, ax=ax2) 
        cbar2.set_label(r'$\sigma_\star$ [km s$^{-1}$]', fontsize=13) 
        cbar2.ax.tick_params(labelsize=12)
        ax2.set_xlabel(r'$\Delta$X [arcsec]', size=12)
        ax2.set_ylabel(r'$\Delta$Y [arcsec]', size=12)

        plt.show()

    ##############################################################################
    if filters is not None:
        if log: log_timing('Getting band-convolved images...')

        # load filter files
        images, image_errs, image_names = [], [], []
        for _filter in filters:
            _filter.convert_units((UNITS[1], '1')) # convert to angstroms

            images.append(np.ma.masked_array(fdc.get_convolved_image(_filter), mask=final_mask))
            image_errs.append(np.ma.masked_array(np.sqrt(((R*fdc)**2).get_convolved_image(_filter)), mask=final_mask))
            image_names.append(_filter.name)

        if plot:
            fig = plt.figure(figsize=(8*len(filters), 5))
            fig.suptitle('Convolved images', fontsize=15)

            for i, _filter in enumerate(filters):
                ax1 = fig.add_subplot(1, len(filters), i+1)
                im1 = plt.imshow(images[i], cmap='gray', extent=extent, origin='lower')
                cbar1 = fig.colorbar(im1, ax=ax1)
                cbar1.set_label(str(image_names[i]) + r' $F_\nu$ [Jy]')
                cbar1.ax.tick_params()
                ax1.set_xlabel(r'$\Delta$X [arcsec]')
                ax1.set_ylabel(r'$\Delta$Y [arcsec]')
            
            plt.show()

            fig = plt.figure(figsize=(8*len(filters), 5))
            fig.suptitle('Error images', fontsize=15)

            for i, _filter in enumerate(filters):
                ax1 = fig.add_subplot(1, len(filters), i+1)
                im1 = plt.imshow(image_errs[i], cmap='gray', extent=extent, origin='lower')
                cbar1 = fig.colorbar(im1, ax=ax1)
                cbar1.set_label(str(image_names[i]) + r' $F_\nu$ err [Jy]')
                cbar1.ax.tick_params()
                ax1.set_xlabel(r'$\Delta$X [arcsec]')
                ax1.set_ylabel(r'$\Delta$Y [arcsec]')
            
            plt.show()

    ##############################################################################
    if log: log_timing('Saving the hdf5 output file at {}...'.format(hf_file))

    with h5py.File(hf_file, 'w') as hf:
        hf.create_dataset('velocity',       data=final_vel_masked.data)
        hf.create_dataset('sigma',          data=final_sigma_masked.data)
        hf.create_dataset('mask',           data=final_mask)
        hf.create_dataset('continous mask', data=final_continous_mask)
        hf.create_dataset('velocity error', data=final_vel_err.data)
        hf.create_dataset('sigma error',    data=final_sigma_err.data)
        if filters is not None:
            for image, image_err, image_name in zip(images, image_errs, image_names):
                hf.create_dataset(str(image_name), data=image.data)
                hf.create_dataset(str(image_name) + ' error', data=image_err.data)
        hf.create_dataset('extent', data=extent)

    if log: log_timing()
    return




if __name__ == "__main__":
    from filearguments import get_filearguments  # custom filearguments file (https://gist.github.com/JaedenBardati/81c4543b84a49584ea09bf529fbdf29c)
    res = get_filearguments(i=str, o=str, require_options=True, option_prefix='-')  # arguments: (input, output) 
    root_files, hf_files = res['-i'], res['-o']
    assert len(root_files) == len(hf_files)
    
    nphotons = 5e9
    psf = GaussianPSF(fwhm=1)
    filters = [
        Filter('/home/jbardati/projects/def-jruan/jbardati/Romulus25/SKIRT/filters/uband.filter', ounits=('micron', '1')), 
        Filter('/home/jbardati/projects/def-jruan/jbardati/Romulus25/SKIRT/filters/gband.filter', ounits=('micron', '1')), 
        Filter('/home/jbardati/projects/def-jruan/jbardati/Romulus25/SKIRT/filters/rband.filter', ounits=('micron', '1'))
    ]
 
    for root_file, hf_file in zip(root_files, hf_files):
        run_ppxf_individual(root_file, hf_file, nphotons, psf=psf, filters=filters, sim_ext='total', parallel=False, progress_bar=False, log=True, plot=False)
        print()
    print('All done!')
