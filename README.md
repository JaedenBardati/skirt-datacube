# skirt-datacube
A repository of convenience python functions and classes that handle the datacube output of the SKIRT radiative transfer simulation.

### Requirements
numpy, matplotlib, scipy, astropy, pandas

### Description
The `fitsdatacube.py` file contains the following convenience functions and classes:

- `FitsDatacube` : Class for loading and accessing the ".fits" file, including contructing and displaying monochromatic images, band-convolved images, pixel spectra and integrated spectra.
- `Filter` : Class for loading and accessing ".filter" files, including interpolating the discrete filter band data for use when convolving datacubes of arbituary wavelengths, and basic plotting routines.
- `PSF` : Class and associated sub-classes (e.g. `GaussianPSF`) for handling PSF convolution on 2d images as well as on datacubes. 
- `load_dat_file` : Function for loading a ".dat" file into a Pandas dataframe. 

I also include the code `run_ppxf_on_skirt.py` which runs ppxf on the SKIRT output, generating line-of-sight velocity and velocity dispersion maps with voronoi binning. The file `extract_kinematic_parameters.py` extracts certain kinematic parameters from these LOSVD maps (this file uses the `nevin_compare_centers.py` and `nevin_radon_python_mod.py` python files, which were taken, with only minor modification, from [Becky Nevin's GitHub](https://github.com/beckynevin/Kinematics_SUNRISE)).

### Examples
See the accompanying Jupyter Notebook `SKIRT-output-access.ipynb` for example usages.
