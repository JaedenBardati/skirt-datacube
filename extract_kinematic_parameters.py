"""
Extracts kinematic parameters on velocity and sigma maps (plus band-convolved images).
Run this file to interface with my ppxf hdf5 file output.

This code packages a few python modules to work. Namely, it uses:
 - statmorph: pip install (Rodriguez-Gomez et al., 2019)
 - Nevin's radon transform code: included in this repo
     - radon_python_mod.py from https://github.com/beckynevin/Kinematics_SUNRISE (Nevin et al., 2019, based on Stark et al. 2018)
     - compare_centers function from https://github.com/beckynevin/Kinematics_SUNRISE/blob/master/Kinematic_Hexagons_Display.ipynb
 - Kinemetry: https://www.aip.de/en/members/davor-krajnovic/kinemetry/ (Krajnovic et al., 2006)
 - pafit (kinemetry): pip install (Krajnovic et al., 2006 including Michele Cappellari)

Jaeden Bardati 2023

Example usage: python "extract_kinematic_parameters.py [input_folder] [output_file]"
  where input_folder contains all the hdf5 containing the LOSVD maps+, and output_file is the csv file to contain kinematic parameter data.
"""

import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import h5py


def extract_LOSVD_data(filename):
    """Extracts the LOSVD data from the given hdf5 file. Change this function if you store the data in another form."""
    data = {}
    with h5py.File(filename, 'r') as hf:        
        # get masks
        final_mask = np.array(hf['mask'])
        if 'continous mask' in hf.keys():
            final_continous_mask = np.array(hf['continous mask'])
        else:
            final_continous_mask = final_mask
            warnings.warn('No continous mask found. Using the other mask.')
        
        # check masks
        if np.shape(final_mask) == (): # whole array is masked or not
            assert not final_mask, 'The entire image is masked!'      # if True
            final_mask = np.zeros(hf['velocity'].shape).astype(bool)  # if False
            warnings.warn('The entire image is unmasked.')
        if np.shape(final_continous_mask) == (): # whole array is masked or not
            assert not final_continous_mask, 'The entire image is masked!'      # if True
            final_continous_mask = np.zeros(hf['velocity'].shape).astype(bool)  # if False
        
        # get other map data
        data['final_mask']           = final_mask
        data['final_continous_mask'] = final_continous_mask
        data['final_vel_masked']     = np.ma.masked_array(hf['velocity'], mask=final_continous_mask)
        data['final_sigma_masked']   = np.ma.masked_array(hf['sigma'], mask=final_continous_mask)
        data['uband_image_masked']   = np.ma.masked_array(hf['uband'], mask=final_continous_mask)
        data['gband_image_masked']   = np.ma.masked_array(hf['gband'], mask=final_continous_mask)
        data['rband_image_masked']   = np.ma.masked_array(hf['rband'], mask=final_continous_mask)
        data['final_vel_err']        = np.ma.masked_array(hf['velocity error'], mask=final_continous_mask)
        data['final_sigma_err']      = np.ma.masked_array(hf['sigma error'], mask=final_continous_mask)
        data['uband_image_err']      = np.ma.masked_array(hf['uband error'], mask=final_continous_mask)
        data['gband_image_err']      = np.ma.masked_array(hf['gband error'], mask=final_continous_mask)
        data['rband_image_err']      = np.ma.masked_array(hf['rband error'], mask=final_continous_mask)
        _arrays = list(data.values())
        assert all(arr.shape == _arrays[0].shape for arr in _arrays[1:]), 'Some of the maps have different shapes!'

        # get non-map data
        data['extent']               = tuple(hf['extent'])
        data['nphotons']             = 5e9 if 'nphotons' not in hf.keys() or hf['nphotons'].shape == () else float(hf['nphotons'][0])  # temp for legacy file support
        data['image_shape']          = final_mask.shape

    assert data['image_shape'][0] == data['image_shape'][1], 'The image is not square.'
    return data


## morphology of general maps
from photutils.aperture import CircularAperture
from scipy.optimize import minimize
from astropy.convolution import convolve, Box2DKernel

def gini(Xdist):
    """Returns Gini index of the distribution."""
    assert len(np.shape(Xdist)) == 1, 'X distribution must be 1d.' 
    X_i = np.sort(Xdist) # order by value
    Xavg = X_i.mean()
    n = len(X_i)
    return ((2*np.arange(n) - n - 1)*np.abs(X_i)).sum()/(abs(Xavg)*n*(n-1))
    
def m20(Xmap, mask=None, ret_cen=True, plot=False, no_error=True):
    """Returns (M20, xc, yc) where xc and yc is the center (in pixels) that minimizes M20. May take some computation time for very large galaxies."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.' 
    if mask is None: 
        mask=np.isnan(Xmap)
    
    # construct 1d sorted flux array
    X_i = Xmap[~mask]
    sorted_args = np.argsort(X_i)[::-1]
    X_i = X_i[sorted_args]
    
    # construct xbin and ybin
    mapshape = np.shape(Xmap)
    y, x = np.mgrid[0:mapshape[0], 0:mapshape[1]]
    xbin = x[~mask] # ravel
    ybin = y[~mask]
    xbin = xbin[sorted_args] # sort
    ybin = ybin[sorted_args]
    
    # find m (index separating the top 20% of brightest pixels)
    Xtot = X_i.sum()
    Xcum = X_i.cumsum()
    m = np.searchsorted(Xcum, 0.2*Xtot) - 1
    if m == -1:
        if no_error:
            m = 0
            warnings.warn('More than 20%% of the brightness is in one pixel.')
        else:
            return -np.inf, None, None   # >20% of the brightness is in one pixel 

    # find xc and yc that minimizes Mtot
    ravel_len = X_i.size
    M_i = np.repeat(np.reshape(X_i, (ravel_len, 1)), ravel_len, axis=1)*(
        (np.repeat(np.reshape(xbin, (ravel_len, 1)), ravel_len, axis=1) - np.repeat(np.reshape(xbin, (1, ravel_len)), ravel_len, axis=0))**2 +
        (np.repeat(np.reshape(ybin, (ravel_len, 1)), ravel_len, axis=1) - np.repeat(np.reshape(ybin, (1, ravel_len)), ravel_len, axis=0))**2
    ) # shape: (different M_i, different xc, yc)
    Mtots = M_i.sum(axis=0) # elements of this vector have different centers
    minarg = np.argmin(Mtots)  # pick center arg with minimum mtot
    Mtot, xc, yc = Mtots[minarg], xbin[minarg], ybin[minarg]
        
    # find m20
    Mcum = M_i.cumsum(axis=0)
    m20 = np.log10(Mcum[m, minarg]/Mtot)
    
    # debug plot
    if plot:
        Mtotsmap=np.ones(mapshape)*np.nan
        for _Mtot, _xc, _yc in zip(Mtots, xbin, ybin):
            Mtotsmap[_yc, _xc] = _Mtot
        plt.title(r'$M_{tot}$ map')
        im = plt.imshow(Mtotsmap, origin='lower', cmap='RdBu_r')
        cbar = plt.colorbar(im, cmap='Reds_r')
        cbar.set_label(r'$M_{tot}$')
        plt.scatter([xc], [yc], marker='x', color='red')
        plt.show()
    
    if not ret_cen:
        return m20
    return m20, xc, yc

def Rmax(Xmap, xc, yc, mask=None):
    """Radius to the maximum distance pixel from the center."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.'
    y, x = np.mgrid[0:np.shape(Xmap)[0], 0:np.shape(Xmap)[1]]
    return np.sqrt(np.ma.masked_array((x-xc)**2 + (y-yc)**2, mask=mask).max())

def RN(Xmap, xc, yc, frac, mask=None, tol=1e-4, x0=1., rmin=None, rmax=None):
    """Circular radius containing N % (ratio) of the flux values."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.'
    assert 0. < frac < 1., 'Frac must be between 0 and 1.'
    
    if rmin is None: rmin = tol
    
    Xtot = Xmap.sum() if mask is None else Xmap[~mask].sum()
    XN = frac*Xtot
    
    def _XN_err(rN_est):
        ap = CircularAperture([xc, yc], rN_est[0])
        XN_est = ap.do_photometry(Xmap, mask=mask)[0][0]
        return (XN_est - XN)**2
    
    res = minimize(_XN_err, x0, bounds=[(rmin, rmax)], tol=tol)  # must be > 0
    assert res.fun < tol, 'R%d did not converge.' % int(100*frac)
    return res.x[0]

def concentration(Xmap, xc, yc, mask=None, ret_radii=False, tol=1e-4, plot=False):
    """Returns the concentration (C)."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.'
    
    rmax = Rmax(Xmap, xc, yc, mask=mask)
    r20 = RN(Xmap, xc, yc, 0.2, mask=mask, tol=tol, rmin=None, rmax=rmax, x0=1.)
    r80 = RN(Xmap, xc, yc, 0.8, mask=mask, tol=tol, rmin=r20, rmax=rmax, x0=(r20+rmax)/2)
    C = 5*np.log10(r80/r20)
    
    if plot:
        plt.figure()
        plt.title('Map (C = %.2f)' % C)
        im = plt.imshow(np.ma.masked_array(Xmap, mask=mask), origin='lower', cmap='magma')
        cbar = plt.colorbar(im, cmap='magma')
        plt.scatter([xc], [yc], marker='x', color='black')
        circle20 = plt.Circle((xc, yc), r20, color='blue', fill=False, label=r'$R_{20}$')
        circle80 = plt.Circle((xc, yc), r80, color='red', fill=False, label=r'$R_{80}$')
        circlemax = plt.Circle((xc, yc), rmax, color='black', fill=False, label=r'$R_{max}$')
        plt.gca().add_patch(circle20)
        plt.gca().add_patch(circle80)
        plt.gca().add_patch(circlemax)
        plt.legend(fontsize=15)
        plt.show()
    
    if ret_radii:
        return C, r20, r80, rmax
    return C
    
def asymmetry(Xmap, xc_init=None, yc_init=None, mask=None, boxwidth=10, noerror=True, ret_cen=True, plot=False, _debug_plot=False):
    """Returns the asymmetry (A) in the form: (A, xc, yc) where xc, yc is the center that minimizes A. 
    Set boxwidth=0 for a fixed xc, yc. """
    shape = np.shape(Xmap)
    assert len(shape) == 2, 'X map must be 2d.'
    assert np.shape(mask) == shape, 'Mask must have the same shape as X map.'
    
    # pad and rotate arrays
    xc_arr, yc_arr = shape[1]//2, shape[0]//2
    if xc_init is None:   # by default use center of image
        xc_init = xc_arr
    if yc_init is None: 
        yc_init = yc_arr
    diff_init_x = xc_init - xc_arr
    diff_init_y = yc_init - yc_arr
    
    # find As in a fixed grid around center
    y, x = np.mgrid[-boxwidth:boxwidth+1, -boxwidth:boxwidth+1]
    ybin, xbin = y.ravel(), x.ravel()
    As = np.ones((2*boxwidth+1)**2)*np.nan
    for i, (diffy, diffx) in enumerate(zip(ybin, xbin)):
        if 0 <= yc_arr+diff_init_y+diffy < mask.shape[0] and 0 <= xc_arr+diff_init_x+diffx < mask.shape[1]:
            if mask is None or mask[yc_arr+diff_init_y+diffy, xc_arr+diff_init_x+diffx] == False:
                # pad to make center of the array the same as the center desired
                extrapadding = ((2*max(0, -diff_init_y-diffy), 2*max(0, diff_init_y+diffy)), (2*max(0, -diff_init_x-diffx), 2*max(0, diff_init_x+diffx)))
                Xpad = np.pad(Xmap, extrapadding, constant_values=np.nan)
                Xpad_mask = np.pad(mask, extrapadding, constant_values=True)
                Xpad = np.ma.masked_array(Xpad, mask=Xpad_mask)  # make masked array

                X180 = np.rot90(Xpad, 2)
                X180_mask = np.rot90(Xpad_mask, 2)
                X180 = np.ma.masked_array(X180, mask=X180_mask)

                X180[ X180_mask & ~Xpad_mask] = 0   # only xpad (assume x180 is zero since its nan)
                Xpad[~X180_mask &  Xpad_mask] = 0   # only x180 (assume xpad is zero since its nan)
                part1A = np.ma.masked_array(Xpad - X180, mask=X180_mask & Xpad_mask)
                part2A = np.ma.masked_array(Xpad, mask=X180_mask & Xpad_mask)

                As[i] = np.abs(part1A).sum()/np.abs(part2A).sum()
            
                if _debug_plot:
                    fig = plt.figure(figsize=(10, 4))
                    fig.suptitle('Iteration %d: A = %.2f' % (i, As[i]), fontsize=20)
                    ax1 = plt.subplot(121)
                    ax1.imshow(Xpad)
                    ax1.scatter(Xpad.shape[1]//2, Xpad.shape[0]//2, marker='x', color='red', label='rotation point')
                    ax2 = plt.subplot(122)
                    ax2.imshow(X180)
                    ax2.scatter(X180.shape[1]//2, X180.shape[0]//2, marker='x', color='red', label='rotation point')
                    plt.show()
        else:
            #warnings.warn('The test box extends outside the image. Ignoring these pixels.')
            pass
    
    # find min A
    argmin = np.nanargmin(As)
    A = As[argmin]
    xc_diff = xbin[argmin]
    yc_diff = ybin[argmin]
    xc = xc_diff + xc_init
    yc = yc_diff + yc_init 
    if not (-boxwidth < xc_diff < boxwidth and -boxwidth < yc_diff < boxwidth):
        if noerror:
            if boxwidth != 1:
                warnings.warn('Asymmetry center is on the boundary of the test grid: Must run again with a new initial center.')
        else:
            raise Exception('Center is on the boundary of the test grid: Must run again with a new initial center.')
    
    if plot:
        plt.figure()
        plt.title('A map')
        im = plt.imshow(As.reshape((boxwidth*2+1, boxwidth*2+1)), extent=(-boxwidth, boxwidth, -boxwidth, boxwidth), origin='lower')
        cbar = plt.colorbar(im, cmap='Reds_r')
        cbar.set_label(r'$A$') 
        plt.scatter([xc_diff], [yc_diff], marker='x', color='red')
        plt.show()

    if not ret_cen:
        return A
    return A, xc, yc

def smoothness(Xmap, mask=None, smooth_size=5):
    """Returns the smoothness (S)."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.'
    
    kernel = Box2DKernel(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore annoying nan convolve warning from astropy for np.nan masked galaxies
        Xsmooth = np.ma.masked_array(convolve(Xmap, kernel, mask=mask), mask=mask)
    Xmap = np.ma.masked_array(Xmap, mask=mask)
    
    return np.abs(Xmap - Xsmooth).sum()/np.abs(Xmap).sum()

def shape_asymmetry(Xmap, xc, yc, mask=None):
    As = asymmetry(Xmap.astype(bool).astype(float), xc_init=xc, yc_init=yc, mask=mask, boxwidth=1, ret_cen=False)
    return As

def find_all_parameters(Xmap, mask=None, xc_init=None, yc_init=None, boxwidth=10, ret_radii=True, ret_As=True, plot=False):
    """Returns all parameters in the form: Gini, M20, (xc_m20, yc_m20), C, A, S, (xc_asym, yc_asym), As."""
    assert len(np.shape(Xmap)) == 2, 'X map must be 2d.'
    if mask is None: 
        mask = np.isnan(Xmap)  # mask nans if no mask is specified
    
    # Gini-M20
    Gini = gini(Xmap[~mask].ravel())
    M20, xc_m20, yc_m20 = m20(Xmap, mask=mask, plot=plot)
    
    # if not initial xc or yc is set, use the mtot min as a first guess 
    if xc_init is None and yc_init is None:
        xc_init = xc_m20
        yc_init = yc_m20
    
    # CAS parameters
    A, xc_asym, yc_asym = asymmetry(Xmap, xc_init, yc_init, boxwidth=boxwidth, mask=mask, plot=plot)
    C, r20, r80, rmax = concentration(Xmap, xc_asym, yc_asym, mask=mask, plot=plot, ret_radii=True)
    S = smoothness(Xmap, mask=mask)
    
    # Shape asymmetry
    if not ret_As:
        if not ret_radii:
            return Gini, M20, (xc_m20, yc_m20), C, A, S, (xc_asym, yc_asym)
        return Gini, M20, (xc_m20, yc_m20), C, (r20, r80, rmax), A, S, (xc_asym, yc_asym)
    
    As = shape_asymmetry(Xmap, xc_asym, yc_asym, mask=mask)
    if not ret_radii:
        return Gini, M20, (xc_m20, yc_m20), C, A, S, (xc_asym, yc_asym), As
    return Gini, M20, (xc_m20, yc_m20), C, (r20, r80, rmax), A, S, (xc_asym, yc_asym), As


def get_morph(image, image_err, psf_kernel=None, npixels=5, sigma_thres=3, reg_size=3, acc=1e-05, maxiter=2000, _print=False, include_doublesersic=True, **kwargs):
    """Statmorph implementation"""
    import statmorph
    import photutils
    import scipy.ndimage as ndi
    
    # get segmentation
    threshold = photutils.detect_threshold(image, sigma_thres)
    segm = photutils.detect_sources(image, threshold, npixels)
    if segm is not None:
        label = np.argmax(segm.areas) + 1  # for now just choose the largest segmentation map
        segmap = np.int64(segm.data == label)
    else:  # if there are no segmentations detected, make the whole thing the segmentation
        label = 1
        segmap = np.int64(np.ones(image.shape))

    # regularize the segmentation
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=reg_size)
    segmap = np.int64(segmap_float > 0.5)

    # run statmorph
    if _print: start = time.time()
    morph = statmorph.source_morphology(image, segmap, weightmap=image_err, psf=psf_kernel, include_doublesersic=include_doublesersic, doublesersic_fitting_args={'acc': acc, 'maxiter': maxiter}, **kwargs)[0]  # include_doublesersic requires v0.5.2
    if _print: print('Time: %g s.' % (time.time() - start))

    return morph


## full extraction

def run_extraction_individual(filename=None, data=None, z=None, pixelscale=None, subvel='mean', 
                              psf=None, n_p=30, n_theta=30, skiplongkin=False, log=False, plot=False):
    """
    Loads LOSVD data from hdf5 files, and extracts various parameters (described in Bardati et al. (in prep)).

    Arguments:

     *(filename)   :   Hdf5 filename to extract data from. Standard format is shown in "extract_LOSVD_data" function 
                       and is made to work easily with my ppxf code for SKIRT simulations.
     *(data)       :   Data to the enter directly (see filename parameter for details).
     +(z)          :   Redshift of the galaxy. Used to get the real distance from the kinematic to the imaging-based centers. 
     +(pixelscale) :   Number of arcseconds in one pixel.
     (subvel)      :   Method to substract the (either None, 'mean' or 'median').
     (psf)         :   PSF used to create band-convolved images (for statmorph).
     (n_p)         :   Number of radii used to make the Radon profile
     (n_theta)     :   Number of angles used to make the Radon profile
     (skiplongkin) :   Whether or not to skip the long kinemetry run.
     (log)         :   Flag to print a log including the timing. 
     (plot)        :   Flag to plots some figures. 
    
    * One of these two arguments are required.
    + These arguments are required for the center comparsion parameters (dist_rband_sigma, dist_rband_vel).
    
    """
    assert filename is not None or data is not None, 'Must enter "filename" to extract data from a file, or the "data" itself.'
    assert filename is None or data is None, 'Can only enter one of either "filename" or "data".'
    assert subvel in ['mean', 'median', 'None', None], 'The "subvel" parameter must be either None, mean or median.'
    
    if log: 
        from my_timing import log_timing
        
    if plot:
        MY_PLT_PARAMS = {'font.size': 24, 'axes.linewidth': 2.0}
        matplotlib.rcParams.update(MY_PLT_PARAMS)
    
    if data is None:
        if log: 
            log_timing('Loading hdf5 data from {}...'.format(filename))
        data = extract_LOSVD_data(filename)
    
    final_mask           = data['final_mask']
    final_continous_mask = data['final_continous_mask']
    final_vel_masked     = data['final_vel_masked']
    final_sigma_masked   = data['final_sigma_masked']
    uband_image_masked   = data['uband_image_masked']
    gband_image_masked   = data['gband_image_masked']
    rband_image_masked   = data['rband_image_masked']
    final_vel_err        = data['final_vel_err']
    final_sigma_err      = data['final_sigma_err']
    uband_image_err      = data['uband_image_err']
    gband_image_err      = data['gband_image_err']
    rband_image_err      = data['rband_image_err']
    extent               = data['extent']
    image_shape          = data['image_shape']

    assert len(extent) == 4, 'Extent must be a tuple of length 4'
    assert len(image_shape) == 2, 'Image size must be a tuple of length 2'
    if np.isnan(final_vel_masked).sum() != 0:
        warnings.warn('The velocity map has NaNs that are not masked.')
    if np.isnan(final_sigma_masked).sum() != 0:
        warnings.warn('The sigma map has NaNs that are not masked.')
    if np.isnan(uband_image_masked).sum() != 0:
        warnings.warn('The u-band image has NaNs that are not masked.')
    if np.isnan(gband_image_masked).sum() != 0:
        warnings.warn('The g-band image has NaNs that are not masked.')
    if np.isnan(rband_image_masked).sum() != 0:
        warnings.warn('The r-band image has NaNs that are not masked.')
    if np.isnan(final_vel_err).sum() != 0:
        warnings.warn('The velocity error map has NaNs that are not masked.')
    if np.isnan(final_sigma_err).sum() != 0:
        warnings.warn('The sigma error map has NaNs that are not masked.')
    if np.isnan(uband_image_err).sum() != 0:
        warnings.warn('The u-band error image has NaNs that are not masked.')
    if np.isnan(gband_image_err).sum() != 0:
        warnings.warn('The g-band error image has NaNs that are not masked.')
    if np.isnan(rband_image_err).sum() != 0:
        warnings.warn('The r-band error image has NaNs that are not masked.')
    
    if subvel == 'mean':
        subtracted_vel = np.mean(final_vel_masked)
        final_vel_masked -= subtracted_vel
    elif subvel == 'median':
        subtracted_vel = np.median(final_vel_masked)
        final_vel_masked -= subtracted_vel
    else:
        subtracted_vel = None
        warnings.warn('The velocity was not mean-subtracted. Some parameters require this.')
    
    if plot:
        # plot out the mask and continous masks
        fig = plt.figure(figsize=(20, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Mask')
        im1 = plt.imshow(final_mask, cmap='seismic', extent=extent, origin='lower', vmin=0, vmax=1)
        ax1.set_xlabel(r'$\Delta$X [arcsec]')
        ax1.set_ylabel(r'$\Delta$Y [arcsec]')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Continous Mask')
        im2 = plt.imshow(final_continous_mask, cmap='seismic', extent=extent, origin='lower', vmin=0, vmax=1)
        ax2.set_xlabel(r'$\Delta$X [arcsec]')
        ax2.set_ylabel(r'$\Delta$Y [arcsec]')

        fig.subplots_adjust(wspace=0.1)  # Adjust the value to change the space between the panels
        fig.subplots_adjust(top=0.85)  # Adjust the value to add space below the title
        plt.show()
        
        
        # plot out the sigma and LOS velocity maps
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('LOSVD moment maps')

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = plt.imshow(final_vel_masked, cmap='RdBu_r', extent=extent, origin='lower')  # viridis, RdBu_r, seismic, coolwarm
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$v_\mathrm{los}$ [km s$^{-1}$]') 
        cbar1.ax.tick_params()
        ax1.set_xlabel(r'$\Delta$X [arcsec]')
        ax1.set_ylabel(r'$\Delta$Y [arcsec]')

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = plt.imshow(final_sigma_masked, cmap='magma', extent=extent, origin='lower')  # viridis, magma, inferno, plasma
        cbar2 = fig.colorbar(im2, ax=ax2) 
        cbar2.set_label(r'$\sigma_\star$ [km s$^{-1}$]') 
        cbar2.ax.tick_params()
        ax2.set_xlabel(r'$\Delta$X [arcsec]')
        ax2.set_ylabel(r'$\Delta$Y [arcsec]')

        fig.subplots_adjust(wspace=0.1)  # Adjust the value to change the space between the panels
        fig.subplots_adjust(top=0.85)  # Adjust the value to add space below the title
        plt.show()


        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('LOSVD moment error maps')

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = plt.imshow(final_vel_err, cmap='RdBu_r', extent=extent, origin='lower')  # viridis, RdBu_r, seismic, coolwarm
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$v_\mathrm{los}$ [km s$^{-1}$]') 
        cbar1.ax.tick_params()
        ax1.set_xlabel(r'$\Delta$X [arcsec]')
        ax1.set_ylabel(r'$\Delta$Y [arcsec]')

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = plt.imshow(final_sigma_err, cmap='magma', extent=extent, origin='lower')  # viridis, magma, inferno, plasma
        cbar2 = fig.colorbar(im2, ax=ax2) 
        cbar2.set_label(r'$\sigma_\star$ [km s$^{-1}$]') 
        cbar2.ax.tick_params()
        ax2.set_xlabel(r'$\Delta$X [arcsec]')
        ax2.set_ylabel(r'$\Delta$Y [arcsec]')

        fig.subplots_adjust(wspace=0.1)  # Adjust the value to change the space between the panels
        fig.subplots_adjust(top=0.85)  # Adjust the value to add space below the title
        plt.show()
    
    
        # look at the band-convolved image maps
        fig = plt.figure(figsize=(25, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        im1 = plt.imshow(uband_image_masked, cmap='Blues_r', extent=extent, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$u$-band  $F_\nu$ [Jy]')
        cbar1.ax.tick_params()
        ax1.set_xlabel(r'$\Delta$X [arcsec]')
        ax1.set_ylabel(r'$\Delta$Y [arcsec]')

        ax2 = fig.add_subplot(1, 3, 2)
        im2 = plt.imshow(gband_image_masked, cmap='Greens_r', extent=extent, origin='lower')
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label(r'$g$-band  $F_\nu$ [Jy]')
        cbar2.ax.tick_params()
        ax2.set_xlabel(r'$\Delta$X [arcsec]')
        ax2.set_ylabel(r'$\Delta$Y [arcsec]')

        ax3 = fig.add_subplot(1, 3, 3)
        im3 = plt.imshow(rband_image_masked, cmap='Reds_r', extent=extent, origin='lower')
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_label(r'$r$-band  $F_\nu$ [Jy]') 
        cbar3.ax.tick_params()
        ax3.set_xlabel(r'$\Delta$X [arcsec]')
        ax3.set_ylabel(r'$\Delta$Y [arcsec]')

        plt.show()

        
        fig = plt.figure(figsize=(25, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        im1 = plt.imshow(uband_image_err, cmap='Blues_r', extent=extent, origin='lower')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$u$-band  $F_\nu$ [Jy]')
        cbar1.ax.tick_params()
        ax1.set_xlabel(r'$\Delta$X [arcsec]')
        ax1.set_ylabel(r'$\Delta$Y [arcsec]')

        ax2 = fig.add_subplot(1, 3, 2)
        im2 = plt.imshow(gband_image_err, cmap='Greens_r', extent=extent, origin='lower')
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label(r'$g$-band  $F_\nu$ [Jy]')
        cbar2.ax.tick_params()
        ax2.set_xlabel(r'$\Delta$X [arcsec]')
        ax2.set_ylabel(r'$\Delta$Y [arcsec]')

        ax3 = fig.add_subplot(1, 3, 3)
        im3 = plt.imshow(rband_image_err, cmap='Reds_r', extent=extent, origin='lower')
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_label(r'$r$-band  $F_\nu$ [Jy]') 
        cbar3.ax.tick_params()
        ax3.set_xlabel(r'$\Delta$X [arcsec]')
        ax3.set_ylabel(r'$\Delta$Y [arcsec]')

        plt.show()
    
    
    # make data dictionary to contain the results
    DATA = {}
    DATA['filename'] = filename
    
    #######################################################
    ## 1d distribution stats
    if log: 
        log_timing('Computing the 1d distribution statistics...')
    
    final_vel_ravel = final_vel_masked.data[~final_vel_masked.mask]
    final_sigma_ravel = final_sigma_masked.data[~final_sigma_masked.mask]

    if plot:
        fig = plt.figure(figsize=(20, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        plt.hist(final_vel_ravel)
        ax1.set_xlabel(r'$v_\mathrm{los}$ [km s$^{-1}$]')

        ax2 = fig.add_subplot(1, 2, 2)
        plt.hist(final_sigma_ravel)
        ax2.set_xlabel(r'$\sigma_\star$ [km s$^{-1}$]')

        plt.show()
        
    # find 1d distribution statistics
    DATA['vel mean'] = np.mean(final_vel_ravel) if subtracted_vel is None else subtracted_vel
    DATA['vel variance'] = np.var(final_vel_ravel)
    DATA['vel skew'] = scipy.stats.skew(final_vel_ravel)
    DATA['vel kurtosis'] = scipy.stats.kurtosis(final_vel_ravel)
    DATA['sig mean'] = np.mean(final_sigma_ravel)
    DATA['sig variance'] = np.var(final_sigma_ravel)
    DATA['sig skew'] = scipy.stats.skew(final_sigma_ravel)
    DATA['sig kurtosis'] = scipy.stats.kurtosis(final_sigma_ravel)


    #######################################################
    ## statmorph on band-convolved images
    if log: 
        log_timing('Computing the morphological properties of the band-convolved images...')
    
    if psf is not None: 
        psf.size = image_shape
        psf = psf.kernel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore annoying background subtraction warning
        uband_morph = get_morph(uband_image_masked.data.copy(), uband_image_err.data.copy(), psf)
        rband_morph = get_morph(rband_image_masked.data.copy(), rband_image_err.data.copy(), psf) 
        gband_morph = get_morph(gband_image_masked.data.copy(), gband_image_err.data.copy(), psf)

    if plot:
        from statmorph.utils.image_diagnostics import make_figure as make_statmorph_figure

        matplotlib.pyplot.rcdefaults()
        make_statmorph_figure(uband_morph)
        plt.show()
        make_statmorph_figure(gband_morph)
        plt.show()
        make_statmorph_figure(rband_morph)
        plt.show()
        matplotlib.rcParams.update(MY_PLT_PARAMS)
    
    extracted_statmorph_parameters = [
        'asymmetry',
        'concentration',
        'deviation',
        'doublesersic_amplitude1',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_amplitude2',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_chi2_dof',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_ellip1',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_ellip2',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_n1',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_n2',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_rhalf1',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_rhalf2',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_theta1',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_theta2',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_xc',  #requires v0.5.2 and include_doublesersic = True
        'doublesersic_yc',  #requires v0.5.2 and include_doublesersic = True
        'ellipticity_asymmetry',
        'ellipticity_centroid',
        'elongation_asymmetry',
        'elongation_centroid',
        'flux_circ',
        'flux_ellip',
        'gini',
        'intensity',
        'm20', 
        'multimode',
        'orientation_asymmetry',
        'orientation_centroid',
        'outer_asymmetry',
        'r20',
        'r50',
        'r80',
        'rhalf_circ',
        'rhalf_ellip',
        'rmax_circ',
        'rmax_ellip',
        'rpetro_circ',
        'rpetro_ellip',
        'sersic_amplitude',
        'sersic_chi2_dof',  #requires v0.5.2
        'sersic_ellip',
        'sersic_n',
        'sersic_rhalf',
        'sersic_theta',
        'sersic_xc',
        'sersic_yc',
        'shape_asymmetry',
        'sky_mean',
        'sky_median',
        'sky_sigma',
        'smoothness',
        'sn_per_pixel',
        'xc_asymmetry',
        'yc_asymmetry',
        'xc_centroid',
        'yc_centroid',
        'flag',
        'flag_sersic',
        'flag_doublesersic',  #requires v0.5.2 and include_doublesersic = True
    ]
    
    for morph_name, morph in zip(['uband', 'gband', 'rband'], [uband_morph, gband_morph, rband_morph]):
        for param in extracted_statmorph_parameters:
            DATA[morph_name + ' ' + param] = morph[param]
    
    #######################################################
    ## Radon profile (using Becky Nevin's code)
    if log: 
        log_timing('Computing the radon profile and kinematic center...')
        
    from nevin_radon_python_mod import radon  # Nevin's code (slightly adapted to work for more general images)
    
    # need to center the velocity at the r-band center
    image_center = (int(rband_morph.yc_asymmetry), int(rband_morph.xc_asymmetry))  # CY, CX in original coords

    extra_shape = (2*(image_center[0] - final_vel_masked.shape[0]//2), 2*(image_center[1] - final_vel_masked.shape[1]//2))
    true_extra_size = max(abs(extra_shape[0]), abs(extra_shape[1]))  # to enforce a square box output
    new_shape = (final_vel_masked.shape[0] + true_extra_size, final_vel_masked.shape[1] + true_extra_size)
    extra2_shape = (true_extra_size - abs(extra_shape[0]), true_extra_size - abs(extra_shape[1]))  # extra shape on extra shape to make it the true extra shape; to make it square
    assert extra2_shape[0] == 0 or extra2_shape[1] == 0, "At least one of the extra2_shape should always be 0"
    assert extra2_shape[0]/2 == extra2_shape[0]//2 and extra2_shape[1]/2 == extra2_shape[1]//2, "The extra2_shape values should always be even"
    extra_slice = (slice(max(0, -extra_shape[0])+extra2_shape[0]//2, min(new_shape[0]-extra_shape[0], new_shape[0])-extra2_shape[0]//2), 
                   slice(max(0, -extra_shape[1])+extra2_shape[1]//2, min(new_shape[1]-extra_shape[1], new_shape[1])-extra2_shape[1]//2))  # for positioning the velocities into the new array shape

    vel_shape_changed = np.ones(new_shape)*np.nan
    vel_shape_changed[extra_slice[0], extra_slice[1]] = final_vel_masked.copy().data
    changed_mask = np.ones(new_shape).astype(bool)
    changed_mask[extra_slice[0], extra_slice[1]] = final_vel_masked.copy().mask
    vel_shape_changed = np.ma.masked_array(vel_shape_changed, mask=changed_mask)
    
    if plot:
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.title.set_text('Original velmap')
        plt.imshow(final_vel_masked, origin='lower', cmap='RdBu_r')
        plt.plot([image_center[1], image_center[1]], [0, final_vel_masked.shape[0]], color='red') #vertical line
        plt.plot([0, final_vel_masked.shape[1]], [image_center[0], image_center[0]], color='red') #horizontal line
        ax1.set_ylim((0, final_vel_masked.shape[0]))
        ax1.set_xlim((0, final_vel_masked.shape[1]))

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.title.set_text('Shape-changed velmap')
        plt.imshow(vel_shape_changed, origin='lower', cmap='RdBu_r')
        plt.plot([vel_shape_changed.shape[1]//2, vel_shape_changed.shape[1]//2], [0, vel_shape_changed.shape[0]], color='red') #vertical line
        plt.plot([0, vel_shape_changed.shape[1]], [vel_shape_changed.shape[0]//2, vel_shape_changed.shape[0]//2], color='red') #horizontal line
        ax2.set_ylim((0, vel_shape_changed.shape[0]))
        ax2.set_xlim((0, vel_shape_changed.shape[1]))

    assert final_vel_masked[image_center[0], image_center[1]].data == vel_shape_changed[vel_shape_changed.shape[0]//2, vel_shape_changed.shape[1]//2].data, 'Something is wrong with the extra shape algorithm and the center of the velocity map is not the image center.'
       
    # run Nevin's code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore some warnings that are annoying
        
        radon_vel = vel_shape_changed.copy()  # to .data or not .data
        
        r_e = rband_morph.rhalf_circ  # use the r-band effective radius
        rad = radon(radon_vel, n_p, n_theta, r_e, 1, 'yes' if plot else 'no')
        if rad[8] == 1:
            expanded = 1
            if log: print('Expanding grid and trying again..')
            rad = radon(radon_vel, n_p, n_theta, r_e, 2, 'yes' if plot else 'no')
            assert rad[8] != 1, 'Must expand grid of centers again. This will increase the uncertainty on the kinematic center, unless a more sofisticated algorithm is made.'
        else:
            expanded = 0
        box_list_min_index, R_AB_list_min_index, radon_A, radon_A2, p_list, theta_list, theta_hat_list_min_index, theta_hat_e_list_min_index, expand = rad
    
    # get relevant parameters
    cx_kin, cy_kin = image_center[1]-box_list_min_index[1], image_center[0]-box_list_min_index[0] # convert back to regular coords
    assert 0 <= cx_kin <= final_vel_masked.shape[1] and 0 <= cy_kin <= final_vel_masked.shape[0], 'Something with the shape-change coordinate transformation went terribly wrong.' 
    
    DATA['radon cx'] = cx_kin
    DATA['radon cy'] = cy_kin
    DATA['radon A'] = radon_A
    DATA['radon A2'] = radon_A2
    DATA['radon expanded'] = expanded
    
    
    #######################################################
    ## Kinemetry 1 (pafit)
    if log: 
        log_timing('Computing the delta PA and residual velocity map...')
    
    from pafit import fit_kinematic_pa as paf  # pip install package

    # prepare data
    kinemetry_center = (final_vel_masked.shape[0]/2, final_vel_masked.shape[1]/2)  # in pixels; should ideally be (cx_kin, cy_kin)

    vel = final_vel_masked.data[~final_vel_masked.mask].ravel().copy()
    vel_err = final_vel_err[~final_vel_masked.mask].ravel().copy()

    vel -= np.median(vel)  # subtract an initial estimate of the systematic velocity

    xbin = ((np.tile(np.arange(0, final_vel_masked.shape[0]),(final_vel_masked.shape[1],1)).ravel() - kinemetry_center[0]))[~final_vel_masked.ravel().mask]
    ybin = ((np.tile(np.arange(0, final_vel_masked.shape[1]).reshape((final_vel_masked.shape[1], 1)), final_vel_masked.shape[0]).ravel()-kinemetry_center[1]))[~final_vel_masked.ravel().mask]
    
    if plot:
        matplotlib.pyplot.rcdefaults()

    # fit the vel model
    angBest, angErr, vSyst = paf.fit_kinematic_pa(xbin, ybin, vel, debug=False, nsteps=361, quiet=True, plot=plot, dvel=vel_err)
    vSyst += np.median(vel)   # add back the initial estimate

    # get the vel model
    vel_model = paf.symmetrize_velfield(xbin, ybin, vel, sym=1, pa=angBest)
    assert vel_model.size == vel.size

    if plot:
        plt.show()
        matplotlib.rcParams.update(MY_PLT_PARAMS)
    
    # find parameters
    resid = np.abs(vel - vel_model).sum()/vel.size
    delta_PA = abs(angBest - (gband_morph.orientation_asymmetry*180/np.pi + 90))  # in deg
    
    DATA['kin angBest'] = angBest
    DATA['kin angErr'] = angErr
    DATA['kin vSyst'] = vSyst
    DATA['kin resid'] = resid
    DATA['kin delta_PA'] = delta_PA
    
    
    #######################################################
    ## Kinemetry 2 (full)
    if not skiplongkin:
        if log: 
            log_timing('Computing the velocity dispersion kinemetry parameters...')
        
        import kinemetry as kin   # python file (see above on where to get)
        from kinemetry import kinemetry 
        
        if plot:
            matplotlib.pyplot.rcdefaults()
        
        try:
            # run kinemetry
            sig = final_sigma_masked.data[~final_vel_masked.mask].ravel().copy()  # this assumes the same mask between sigma and vel
            sig_err = final_sigma_masked[~final_vel_masked.mask].ravel().copy()
            k = kinemetry(xbin, ybin, sig, error=sig_err, fixcen=False, scale=0.1, nrad=300, plot=plot, verbose=False)
            
            if plot:
                matplotlib.rcParams.update(MY_PLT_PARAMS)
            
            # find parameters
            sigma_asym = np.mean((k.cf[:, 1:-1]/5).sum(axis=1)/k.cf[:, 0])
            vel_asym = np.mean((k.cf[:, 2:-1]/4).sum(axis=1)/k.cf[:, 1])

            DATA['kin sigma_asym'] = sigma_asym
            DATA['kin vel_asym'] = vel_asym
        except:
            DATA['kin sigma_asym'] = np.nan
            DATA['kin vel_asym'] = np.nan
            warnings.warn('Long kinemetry run errored, filling sigma_asym and vel_asym with NaNs.')
    
    
    #######################################################
    ## Spin parameter
    if log: 
        log_timing('Computing the spin parameter...')
    
    _xs = np.tile((np.arange(0, final_vel_masked.shape[0]) - cx_kin),(final_vel_masked.shape[1],1))
    _ys = np.tile((np.arange(0, final_vel_masked.shape[1]) - cy_kin).reshape((final_vel_masked.shape[1], 1)), final_vel_masked.shape[0])
    dmap_kin = np.sqrt(_xs**2 + _ys**2)

    spin_param = (rband_image_masked*dmap_kin*np.abs(final_vel_masked)).sum()/(rband_image_masked*dmap_kin*(final_vel_masked**2+final_sigma_masked**2)**0.5).sum()
    fast_rotator_stat = spin_param - 0.08 - rband_morph.ellipticity_asymmetry/4   # use rband ellipticity

    DATA['my spin_param'] = spin_param
    DATA['my fast_rotator_stat'] = fast_rotator_stat # fast rotator if > 0
    
    
    #######################################################
    ## Compare centers (using Nevin's code)
    if z is None or pixelscale is None:
        warnings.warn('Skipping over center comparison since redshift and/or pixelscale were not entered.')
    else:
        if log: 
            log_timing('Computing the distance between kinematic and imaging centers...')

        from nevin_compare_centers import compare_centers  # nevin's code

        delta_x_vel, delta_x_sig  = compare_centers(rband_image_masked, final_sigma_masked, cx_kin, cy_kin, z, pixelscale)
        DATA['nevin delta_x_vel'] = delta_x_vel
        DATA['nevin delta_x_sig'] = delta_x_sig   # distance between centers in kpc
    
    
    #######################################################
    ## Morphology of the kinematic maps
    if log: 
        log_timing('Computing the morphologies of the kinematic maps...')
    
    vel_Gini, vel_M20, (vel_xc_m20, vel_yc_m20), vel_C, (vel_r20, vel_r80, vel_rmax), vel_A, vel_S, (vel_xc_asym, vel_yc_asym), vel_As = find_all_parameters(np.abs(final_vel_masked.data - np.mean(final_vel_masked)), mask=final_mask, ret_As=True, plot=plot) # shape asymmetry on vel will be the same as sigma, so there is no need to do it twice
    sig_Gini, sig_M20, (sig_xc_m20, sig_yc_m20), sig_C, (sig_r20, sig_r80, sig_rmax), sig_A, sig_S, (sig_xc_asym, sig_yc_asym), sig_As = find_all_parameters(final_sigma_masked.data, mask=final_mask, ret_As=True, plot=plot)
    vel_contAs = shape_asymmetry(np.abs(final_vel_masked.data - np.mean(final_vel_masked)), vel_xc_asym, vel_yc_asym, mask=final_continous_mask)
    sig_contAs = shape_asymmetry(final_sigma_masked.data, vel_xc_asym, vel_yc_asym, mask=final_continous_mask)

    DATA['vel Gini'] = vel_Gini
    DATA['vel M20'] = vel_M20
    DATA['vel xc_m20'] = vel_xc_m20
    DATA['vel yc_m20'] = vel_yc_m20
    DATA['vel C'] = vel_C
    DATA['vel r20'] = vel_r20
    DATA['vel r80'] = vel_r80
    DATA['vel rmax'] = vel_rmax
    DATA['vel A'] = vel_A
    DATA['vel S'] = vel_S
    DATA['vel xc_asym'] = vel_xc_asym
    DATA['vel yc_asym'] = vel_yc_asym
    DATA['vel As'] = vel_As
    DATA['vel cont_As'] = vel_contAs
    DATA['sig Gini'] = sig_Gini
    DATA['sig M20'] = sig_M20
    DATA['sig xc_m20'] = sig_xc_m20
    DATA['sig yc_m20'] = sig_yc_m20
    DATA['sig C'] = sig_C
    DATA['sig r20'] = sig_r20
    DATA['sig r80'] = sig_r80
    DATA['sig rmax'] = sig_rmax
    DATA['sig A'] = sig_A
    DATA['sig S'] = sig_S
    DATA['sig xc_asym'] = sig_xc_asym
    DATA['sig yc_asym'] = sig_yc_asym
    DATA['sig As'] = sig_As
    DATA['sig cont_As'] = sig_contAs
    
    #######################################################
    ## Diagnostics
    if log:
        log_timing()

    DATA['image sidelength'] = image_shape[0]
    DATA['image nphotons']  = data['nphotons']
    DATA['image ngoodpix']  = (final_mask == False).sum()  # number of pixels not masked
    DATA['image ncontinousgoodpix']  = (final_continous_mask == False).sum()  # number of pixels not continous masked

    return DATA

        
## Defaults
from fitsdatacube import GaussianPSF # custom package

DEFAULT_PSF = GaussianPSF(fwhm=1) ## this is the same one that is entered into run_ppxf_individual in run_ppxf_on_skirt.py 
DEFAULT_PIXELSCALE = 0.1
DEFAULT_FILEEXT = '.h5'

def DEFAULT_REDSHIFT_FUNCTION(filename):
    """This must be customized. This specfic function is unique to romulus25 formatted in a way that I like."""
    import os
    just_filename = os.path.split(filename)[1]  # get just the filename from the path
    step = int(just_filename[6:].split('_')[0]) # custom identifier form
    step_redshift_dictionary = {
        694: 4.977045553983204, 909: 3.9987908900057905, 1270: 2.999730532662382, 1945: 1.999866242206278, 
        2048: 1.8960673164146993, 2159: 1.7932867461336923, 2281: 1.689754445252182, 2304: 1.6712331299362209, 
        2411: 1.588839351771342, 2536: 1.4997393169017066, 2547: 1.492236174108259, 2560: 1.4834355967903856, 
        2690: 1.3992051629582094, 2816: 1.3235575219650384, 2840: 1.309758252352847, 2998: 1.2233366039282516, 
        3072: 1.1853049846852821, 3163: 1.140479124986304, 3328: 1.0641819980905391, 3336: 1.0606344086283346, 
        3478: 0.9998100146757658, 3517: 0.9837816305553304, 3584: 0.9568846984293302, 3707: 0.9094935825547465, 
        3840: 0.8609266989851763, 3905: 0.8381287532902209, 4096: 0.774398375005892, 4111: 0.769587321565466, 
        4173: 0.7499841583835849, 4326: 0.7034631603233255, 4352: 0.6958070914669776, 4549: 0.6399790373658016, 
        4608: 0.6239681449700463, 4781: 0.5787550553108292, 4864: 0.5579284854772784, 5022: 0.5197198785548629, 
        5107: 0.4999001078658487, 5120: 0.4969122099805612, 5271: 0.4630143029385907, 5376: 0.44028086756423757, 
        5529: 0.40830521105884254, 5632: 0.3875040573920594, 5795: 0.3557019492001796, 5888: 0.338137317038685, 
        6069: 0.30508457947303547, 6144: 0.29180526368008475, 6350: 0.25650758630744686, 6390: 0.24984487724165594, 
        6400: 0.24818857919248094, 6640: 0.20952035371375488, 6656: 0.20701384803795775, 6912: 0.1680455397893872, 
        6937: 0.16435041362805203, 7168: 0.13107962307367194, 7212: 0.1249144540624576, 7241: 0.12088008658345895, 
        7394: 0.09996738303039998, 7424: 0.09593843407808444, 7552: 0.07900276297443654, 7680: 0.06246651955101412, 
        7779: 0.04994019580335607, 7869: 0.03874529195980081, 7936: 0.030527243817950023, 8192: -7.465139617579553e-12
    } # unique to romulus25
    redshift = step_redshift_dictionary[step]
    return redshift


## main 
def main(input_folder, output_file, redshift_function=DEFAULT_REDSHIFT_FUNCTION, shuffle_folder=False,
         ext=DEFAULT_FILEEXT, psf=DEFAULT_PSF, pixelscale=DEFAULT_PIXELSCALE, log=False, plot=False, raise_error=True):
    """Runs parameter extraction on all files in a folder specified by the file arguments.
    Redshift function takes a filename in the folder and determines the redshift from it."""
    import glob, os
    import pandas as pd
 
    if log: 
        from my_timing import log_timing

    # construct dataframe
    data = []
    glob_of_files = glob.glob(os.path.join(input_folder, '*' + ext))
    if shuffle_folder:
        import numpy as np
        glob_of_files = sorted(glob_of_files, key=lambda k: np.random.rand())

    if log:
        print('There are {} files in the folder {} .\n'.format(len(glob_of_files), input_folder))
    
    for filename in glob_of_files:
        if log: 
            log_timing('Extracting parameters from {} ...'.format(filename))
        try:
            data.append(run_extraction_individual(filename=filename, z=redshift_function(filename), 
                                                  pixelscale=pixelscale, psf=psf, log=False, plot=plot))
        except Exception as e:
            if raise_error:
                raise e
            else:
                warnings.warn(str(e))
    
    if log: 
        log_timing()

    df = pd.DataFrame(data)
    
    # save df to csv
    df.to_csv(output_file, index=False, header=True)
    if log: 
        print('All done!')



if __name__ == "__main__":
    from filearguments import get_filearguments  # custom filearguments file (https://gist.github.com/JaedenBardati/81c4543b84a49584ea09bf529fbdf29c)
    
    # get file arguments
    res = get_filearguments(str, str)  # arguments: (input folder, output csv file)
    input_folder, output_file = res
    
    # run the main program using defaults
    matplotlib.use('Agg')
    main(input_folder, output_file, log=True, plot=False, raise_error=False)

  
