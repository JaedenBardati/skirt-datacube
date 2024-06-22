"""
This python file contains classes to extract the datacube output from the '.fits' files of SKIRT. 
It also extracts data from a '.dat' file integrated spectrum output from SKIRT.

Use the class FitsDatacube for loading and accessing the fits file, including contructing and displaying monochromatic images, band-convolved images, pixel spectra and integrated spectra.
The accompanying Filter class loads and accesses '.filter' files, along with interpolating the discrete filter band data for use when convolving datacubes of arbituary wavelengths, as well as some basic plotting routines.
Use the `load_dat_file` function to load a dat file into a Pandas DataFrame. 

See the accompanying Jupyter Notebook SKIRT-output-access.ipynb for example usages.

REQUIRES: numpy, matplotlib, scipy, astropy, pandas (for load_dat_file only)

Jaeden Bardati 2023
"""

from abc import ABCMeta
from os import path
import warnings

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d as scipy_interp1d
from scipy.signal import convolve as scipy_convolve

from astropy import units as astropy_u
from astropy.io import fits as astropy_fits
from astropy.utils.decorators import lazyproperty


class FitsDatacube:
    """
        Class that wraps some useful methods for loading and accessing a fits image.
    """
    DEFAULT_UNITS = ('Jy', 'micron')
    
    def _load(self, iunits=DEFAULT_UNITS, ounits=DEFAULT_UNITS):
        # Loads a fits file at <filename>.
        with astropy_fits.open(self.filename) as _hdu_list:
            assert len(_hdu_list) == 2, 'Unrecognized fits datacube format. The <FitsDatacube> requires two HDUs: the first must be a PrimaryHDU containing the datacube and the second must be a TableHDU containing the wavelengths.'
            self._dc = np.array(_hdu_list[0].data, dtype=float)
            self.wav = np.array(_hdu_list[1].data, dtype=float)
        
        assert len(self._dc.shape) == 3 and len(self.wav.shape) == 1, 'Unrecognized fits datacube format. The first HDU must be the datacube and the second TableHDU must be the wavelengths.'
        self.shape = self._dc.shape
        self._nwav = self._dc.shape[0]
        self._npix = self._dc.shape[1:]
        assert self._nwav == len(self.wav), 'Unrecognized fits datacube format. The wavelengths must correspond to the zeroth axis of the datacube.'
        if self._npix[0] != self._npix[1]: 
            warnings.warn('Non-square datacube. You may encounter some unexpected issues.')
        
        self.units = iunits
        if iunits != ounits: 
            self.convert_units(ounits)
    
    def __init__(self, filename=None, iunits=DEFAULT_UNITS, ounits=DEFAULT_UNITS, _no_load=False):
        """
        Loads a fits file at <filename>.
        <iunits> indicate the input units (fits file) and <ounits> indicate the output units (FitsDatacube instance)
        Units lists have the form: [image units, wavelength units].
        """
        if not _no_load:
            self.filename = filename
            self._load(iunits=iunits, ounits=ounits)
        else:
            self.filename = None
            self._dc = None
            self.wav = None
            self.shape = None
            self._nwav = None
            self._npix = None
            self.units = None
    
    def __getitem__(self, index):
        return self._dc[index]
    
    def __setitem__(self, index, newvalue):
        self._dc[index] = newvalue

    def copy(self, _dc_operation=None):
        _fdc = FitsDatacube(_no_load=True)
        _fdc.filename = self.filename
        _fdc._dc = self._dc.copy() if _dc_operation is None else _dc_operation(self._dc)
        _fdc.wav = self.wav.copy()
        _fdc.shape = self.shape
        _fdc._nwav = self._nwav
        _fdc._npix = self._npix
        _fdc.units = self.units
        return _fdc
    
    __add__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__add__(y if type(y) is not FitsDatacube else y._dc))
    __sub__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__sub__(y if type(y) is not FitsDatacube else y._dc))
    __mul__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__mul__(y if type(y) is not FitsDatacube else y._dc))
    __truediv__  = lambda self, y: self.copy(_dc_operation=lambda x: x.__truediv__(y if type(y) is not FitsDatacube else y._dc))
    __pow__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__pow__(y if type(y) is not FitsDatacube else y._dc))
    __pos__      = lambda self: self.copy(_dc_operation=lambda x: x.__pos__())
    __neg__      = lambda self: self.copy(_dc_operation=lambda x: x.__neg__())
    
    __radd__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__radd__(y if type(y) is not FitsDatacube else y._dc))
    __rsub__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__rsub__(y if type(y) is not FitsDatacube else y._dc))
    __rmul__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__rmul__(y if type(y) is not FitsDatacube else y._dc))
    __rtruediv__  = lambda self, y: self.copy(_dc_operation=lambda x: x.__rtruediv__(y if type(y) is not FitsDatacube else y._dc))
    __rpow__      = lambda self, y: self.copy(_dc_operation=lambda x: x.__rpow__(y if type(y) is not FitsDatacube else y._dc))
    
    __eq__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__eq__(y if type(y) is not FitsDatacube else y._dc))
    __ne__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__ne__(y if type(y) is not FitsDatacube else y._dc))
    __lt__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__lt__(y if type(y) is not FitsDatacube else y._dc))
    __le__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__le__(y if type(y) is not FitsDatacube else y._dc))
    __gt__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__gt__(y if type(y) is not FitsDatacube else y._dc))
    __ge__        = lambda self, y: self.copy(_dc_operation=lambda x: x.__ge__(y if type(y) is not FitsDatacube else y._dc))
    
    __and__       = lambda self, y: self.copy(_dc_operation=lambda x: np.logical_and(x, y._dc) if type(y) is FitsDatacube else x.__and__(y))  
    __or__        = lambda self, y: self.copy(_dc_operation=lambda x: np.logical_or(x, y._dc) if type(y) is FitsDatacube else x.__or__(y))
    __invert__    = lambda self: self.copy(_dc_operation=lambda x: x.__invert__())

    log10 = lambda self, *args, **kwargs: self.copy(_dc_operation=lambda x: np.log10(x, *args, **kwargs))
    log = lambda self, *args, **kwargs: self.copy(_dc_operation=lambda x: np.log(x, *args, **kwargs))
    
    def sum(self): return np.sum(self._dc)
    def max(self): return np.max(self._dc)
    def min(self): return np.min(self._dc)
    def mean(self): return np.mean(self._dc)
    def median(self): return np.median(self._dc)
    def std(self): return np.std(self._dc)
    
    def convert_units(self, ounits):
        """Converts the unit system."""
        self._dc *= astropy_u.Unit(self.units[0]).in_units(ounits[0])
        self.wav *= astropy_u.Unit(self.units[1]).in_units(ounits[1])
        self.units = ounits
    
    def wav_index(self, wav):
        """Returns the wavelength index nearest to the inputted wavelength."""
        return np.argmin(np.abs(wav - self.wav))
    
    def infer_R(self, thres=0.01):
        """Infers the spectral_resolution from the given wavelengths."""
        R1 = self.wav[:-1]/np.diff(self.wav)
        R2 = self.wav[1:]/np.diff(self.wav)
        if np.any(np.diff(R1) > thres*R1[1:]) or np.any(np.diff(R2) > thres*R2[:-1]):
            raise Exception("Trouble inferring spectral resolution. Maybe the wavelengths aren't logarithmically distributed?")
        return (np.mean(R1) + np.mean(R2))/2
    
    def convolve_PSF(self, psf, inplace=False):
        if PSF not in type(psf).__mro__:
            psf = CustomPSF(psf)
        copy = self if inplace else self.copy()
        copy._dc = psf._convolve_over(copy._dc)
        return copy

    def get_image(self, wav):
        """Returns the 2d image at a given wavelength."""
        return self[self.wav_index(wav), :, :]

    def get_spectrum(self, x, y):
        """Returns an array containing the spectrum values, given the pixel-space 
        position (x, y), with (0, 0) centered in the lower left corner. """
        return self[:, x, y]
    
    def get_convolved_image(self, filt):
        """Returns the 2d image for a given filter (throughput function of wavelength)."""
        image = np.zeros(self._npix)
        for wav in self.wav:
            image += self.get_image(wav) * filt(wav)
        return image
    
    def get_integrated_spectrum(self, filt2d=None):
        """Returns the spectrum for a given 2d filter (weight function of position).
           If no filter is entered, the whole image is integrated with an equal and 
           normalized weighting."""
        if filt2d is None:
            filt2d = lambda x, y: 1/(self._npix[0]*self._npix[1])
            
        spectrum = np.zeros(self._nwav)
        for x in range(self._npix[0]):
            for y in range(self._npix[1]):
                spectrum += self.get_spectrum(x, y) * filt2d(x, y)
        return spectrum
    
    @staticmethod
    def _general_plot(subplot=None, figsize=(8, 6), xscale='linear', yscale='linear', 
                      title=None, xlim=None, ylim=None, xlabel=None, ylabel=None):
        if subplot is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = subplot
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        return (fig, ax)
    
    @staticmethod
    def _plot_image(image, cmap='gray', logscale=False, vmin=None, vmax=None, ret_im=False, **kwargs):
        fig, ax = FitsDatacube._general_plot(**kwargs)
        if logscale:
            image = np.log10(image.copy())
        im = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        return (fig, ax) if not ret_im else (fig, ax, im)
    
    @staticmethod
    def _plot_spectrum(wavs, thrus, color=None, alpha=None, **kwargs):
        fig, ax = FitsDatacube._general_plot(**kwargs)
        ax.plot(wavs, thrus, color=color, alpha=alpha)
        return (fig, ax)
    
    def plot_image(self, wav, **kwargs):
        return self._plot_image(self.get_image(wav), **kwargs)
    
    def plot_convolved_image(self, filt, **kwargs):
        return self._plot_image(self.get_convolved_image(filt), **kwargs)
    
    def plot_spectrum(self, x, y, **kwargs):
        return self._plot_spectrum(self.wav, self.get_spectrum(x, y), **kwargs)
    
    def plot_integrated_spectrum(self, filt2d=None, **kwargs):
        return self._plot_spectrum(self.wav, self.get_integrated_spectrum(filt2d=filt2d), **kwargs)

    def plot_simpleRGB(self, Rfilt, Gfilt, Bfilt, rprefoo=None, gprefoo=None, bprefoo=None, rpostfoo=None, gpostfoo=None, bpostfoo=None, **kwargs):
        rimage, gimage, bimage = self.get_convolved_image(Rfilt), self.get_convolved_image(Gfilt), self.get_convolved_image(Bfilt)
        if rprefoo is not None: rimage = rprefoo(rimage)
        if gprefoo is not None: gimage = gprefoo(gimage)
        if bprefoo is not None: bimage = bprefoo(bimage)
        rimage, gimage, bimage = rimage/rimage.max(), gimage/gimage.max(), bimage/bimage.max()   # normalize
        if rpostfoo is not None: rimage = rpostfoo(rimage)
        if gpostfoo is not None: gimage = gpostfoo(gimage)
        if bpostfoo is not None: bimage = bpostfoo(bimage)
        rgb_image = np.dstack((rimage, gimage, bimage))
        return self._plot_image(rgb_image, cmap=None, **kwargs)
    
    
class Filter:
    """
    Class that wraps loading and accessing filter files.
    """
    DEFAULT_DELIM = ' '
    DEFAULT_KIND = 'linear'
    DEFAULT_UNITS = ('micron', '1')  # wav, thru
    
    def __init__(self, filename, delim=DEFAULT_DELIM, iunits=DEFAULT_UNITS, ounits=DEFAULT_UNITS, interp_kind=DEFAULT_KIND, _no_load=False):
        """
        Loads a filter file at <filename>.
        """
        self.filename = filename
        self.name = path.splitext(path.basename(filename))[0] if filename is not None else None
        if not _no_load: 
            self.wav, self.thru = self._load(delim=delim)
        self.interp_kind = interp_kind
        
        self.units = iunits
        if iunits != ounits: 
            self.convert_units(ounits)
       
    def __call__(self, wav):
        return self.filt(wav)
    
    def copy(self):
        _filt = Filter(self.filename, _no_load=True)
        _filt.wav = self.wav.copy()
        _filt.thru = self.thru.copy()
        _filt.interp_kind = self.interp_kind
        _filt.units = self.units
        return _filt

    def _load(self, delim=DEFAULT_DELIM):
        arr = np.genfromtxt(self.filename, delimiter=delim).T
        return arr[0], arr[1]
        
    @lazyproperty
    def filt(self):
        """
        Creates an interpolated filter function of the wavelength. 
        """
        return scipy_interp1d(self.wav, self.thru, kind=self.interp_kind, bounds_error=False, fill_value=0)
    
    def convert_units(self, ounits):
        self.wav *= astropy_u.Unit(self.units[0]).in_units(ounits[0])
        self.thru *= astropy_u.Unit(self.units[1]).in_units(ounits[1])
        self.units = ounits
    
    def plot(self, **kwargs):
        return FitsDatacube._plot_spectrum(self.wav, self.thru, **kwargs)
    
    def plot_interpolated(self, wavmin, wavmax, num=100, **kwargs):
        wavs = np.linspace(wavmin, wavmax, num=num)
        return FitsDatacube._plot_spectrum(wavs, self(wavs), **kwargs)
    

class BinaryFilter(Filter):
    DEFAULT_KIND = None

    def __init__(self, minwav, maxwav, val=1.0, num=100, interp_kind=DEFAULT_KIND):
        super().__init__(None, interp_kind=interp_kind, _no_load=True)
        self.name = 'Binary Filter'
        self.wav = np.linspace(minwav, maxwav, num=num)
        self.thru = np.ones(num)*val
        self.minwav = minwav
        self.maxwav = maxwav
        self.val = val

    @lazyproperty
    def filt(self):
        if self.interp_kind is None:
            return np.vectorize(lambda x: (self.val if self.minwav < x and x < self.maxwav else 0), otypes=[np.float64])
        else:
            return super().filt



class PSF(metaclass=ABCMeta):
    """
    Metaclass that constructs a PSF for convolving over a 2d image.
    """
    def __init__(self, size=None):
        self.size = size
        if size is not None:
            self.kernel   # pre load kernel if construction is possible

    @lazyproperty
    def kernel(self):
        """The kernel that is convolved with over the image."""
        if self.size is None: raise Exception('Kernel size must first be set with <PSF.size>.')
        return self._construct_kernel()

    def _construct_kernel(self):
        raise NotImplementedError  # must be implemented in the psf child classes

    def _convolve_over(self, image):
        if len(image.shape) != 2 and len(image.shape) != 3: raise ValueError('Image must be either a 2d or 3d object.')
        if self.size is None: 
            self.size = image.shape[-2:]  # take the last 2 dimensions as the image shape to use 
        kernel = self.kernel
        if len(image.shape) == 3:
            kernel = kernel.reshape([1, *kernel.shape])
        return scipy_convolve(image, kernel, mode='same')


class GaussianPSF(PSF):
    """
    Class that constructs a Gaussian PSF using a fwhm or sigma.
    """
    def __init__(self, fwhm=None, sigma=None, size=None):
        # Must enter either fwhm or sigma (in pixels). If size is None, it will use the full size of the image it is convolved over.
        if fwhm is None and sigma is None: raise ValueError('Must specify either FWHM or sigma for the Gaussian distribution.')
        if fwhm is not None and sigma is not None: raise ValueError('Must only specify FWHM or sigma for the Gaussian distribution, not both.')
        self.sigma = sigma if sigma is not None else fwhm/2.35482004503   # 2*sqrt(2ln2)
        super().__init__(size=size)

    def _construct_kernel(self):
        if hasattr(self.size, '__getitem__'):
            sizeX = self.size[0]//2
            sizeY = self.size[1]//2
        else:
            sizeX = self.size//2
            sizeY = self.size//2
        y, x = np.mgrid[-sizeY:sizeY+1, -sizeX:sizeX+1]
        psf = np.exp(-(x*x + y*y)/(2.0*self.sigma**2))
        psf /= np.sum(psf)
        return psf
    
class CustomPSF(PSF):
    """Custom class to incorporate a custom array psf."""
    def __init__(self, kernel):
        try:
            size=kernel.shape
            assert len(size) == 2 or len(size) == 3, 'The kernel must be a 2d or 3d array.'
        except AttributeError:
            raise TypeError('The kernel must have a shape attribute.')
        self.kernel = kernel
        super().__init__(size=size)


def load_dat_file(filename):
    """Function that loads a .dat file in the format of SKIRT input/output."""
    # get header
    import pandas as pd

    header = {}
    firstNonCommentRowIndex = None
    with open(filename) as file:
        for i, line in enumerate(file):
            l = line.strip()
            if l[0] == '#':
                l = l[1:].lstrip()
                if l[:6].lower() == 'column':
                    l = l[6:].lstrip()
                    split_l = l.split(':')
                    assert len(split_l) == 2 # otherwise, unfamiliar form!
                    icol = int(split_l[0]) # error here means we have the form: # column %s, where %s is not an integer
                    l = split_l[1].lstrip() # this should be the column name
                    header[icol] = l
            else:
                firstNonCommentRowIndex = i
                break
    assert firstNonCommentRowIndex is not None # otherwise the entire file is just comments
    
    # get data
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=firstNonCommentRowIndex, header=None)
    
    # adjust column names
    if firstNonCommentRowIndex == 0:
        columns = None
    else:
        columns = [None for i in range(max(header.keys()))]
        for k, v in header.items(): columns[k-1] = v
        assert None not in columns # otherwise, missing column 
        df.columns = columns
    
    return df
