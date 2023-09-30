import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import ndimage
import cv2

def gauss(x,a,x0,sigma,offset):
    return a*exp(-(x-x0)**2/(2*sigma**2))+offset
            
def fit_2_gaussian(data):
    """
    Fits 2D gaussians to surface brightness using the guesses from the low pass filter of the galaxy locations
    Basically, this is my own design of a rough precursor to Source Extractor
    """
    # Create x and y indices
    data=np.flipud(data)
    x = np.linspace(0, np.shape(data)[0]-1,np.shape(data)[0])
    y = np.linspace(0, np.shape(data)[1]-1, np.shape(data)[1])
    x, y = np.meshgrid(x, y)

    plt.clf()
    plt.close()
    fig, ax = plt.subplots(1, 1)
    #ax.hold(True)
    #im = ax.imshow(data, cmap=plt.cm.jet, origin='bottom')
    #I haven't figured out how to create contours without also creating a plot
    cs = ax.contour(x, y, data, 8, colors='w')
    plt.clf()
    plt.close()
    
    p = cs.levels#collections[0].get_paths()[0]
    
    #Now, snatch the last level and use it to blur everything else out and find the center with a binary threshold
    ret,thresh = cv2.threshold(data,p[-1]/2,2000,cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    
    # calculate x,y coordinate of center
    
    cX = int(M["m10"] / M["m00"])   
    cY = int(M["m01"] / M["m00"])
    
    return cX, np.shape(data)[0]-cY



def compare_centers(image, disp, x_kin, y_kin, z, pixel_scale):
    """
    Compares the centeroid of the r-band image to that of the velocity dispersion 2D map,
    also to the kinematic center (x_kin,y_kin).
    It does this in terms of a physical distance in kpc, given the redshift z of the galaxy.
    pixel_scale is the arcseconds in one pixel.
    """
    
    #Apply a 10x10 kernal to the image to filter out noise (its basically a low pass filter)
    #to smooth things out
    kernel = np.ones((10,10))

    lp = ndimage.convolve(image.filled(fill_value=0), kernel)#was result
    
    c=fit_2_gaussian(lp)

    img_cen_x = c[0]
    img_cen_y = c[1]
    
    #Do this whole thing again but for the velocity dispersion map
    kernel = np.ones((10,10))

    lp = ndimage.convolve(disp.filled(fill_value=0), kernel)#was result

    c=fit_2_gaussian(lp)

    disp_cen_x = c[0]
    disp_cen_y = c[1]
    
    
    kpc_arcsec=(cosmo.kpc_proper_per_arcmin(z).value/60)
    spax_to_kpc = pixel_scale*kpc_arcsec
    
    return spax_to_kpc*np.sqrt((img_cen_x-disp_cen_x)**2+(img_cen_y-disp_cen_y)**2), spax_to_kpc*np.sqrt((img_cen_x-x_kin)**2+(img_cen_y-y_kin)**2)

