# skirt-datacube
A repository of convenience python functions and classes that handle the datacube output of the SKIRT radiative transfer simulation.

### Requirements
numpy, matplotlib, scipy, astropy, pandas

### Description
The `fits_datacube.py` file contains the following convenience functions and classes:

- `FitsDatacube` : Class for loading and accessing the ".fits" file, including contructing and displaying monochromatic images, band-convolved images, pixel spectra and integrated spectra.
- `Filter` : Class for loading and accessing ".filter" files, including interpolating the discrete filter band data for use when convolving datacubes of arbituary wavelengths, and basic plotting routines.
- `PSF` : Class and associated sub-classes (e.g. `GaussianPSF`) for handling PSF convolution on 2d images as well as on datacubes. 
- `load_dat_file` : Function for loading a ".dat" file into a Pandas dataframe. 

### Examples
See the accompanying Jupyter Notebook `SKIRT-output-access.ipynb` for example usages.
