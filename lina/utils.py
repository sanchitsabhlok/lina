from .math_module import xp, _scipy, ensure_np_array
from . import imshows

import numpy as np
import scipy
import astropy.units as u

import poppy

from astropy.io import fits
import pickle

import poppy

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def rotate_arr(arr, rotation, reshape=False, order=1):
    if arr.dtype == complex:
        arr_r = _scipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = _scipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = _scipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(arr, pixelscale, new_pixelscale, order=1):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                             -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = xp.array([ivals, jvals])

        interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

def generate_wfe(diam, wavelength=500*u.nm,
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05,
                 npix=256, oversample=4,  
                 plot=False):
    
    amp_rms *= u.nm
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms, radius=diam/2, seed=amp_seed).get_opd(wf)
    wfe_amp /= amp_rms.unit.to(u.m)
    amp_rms = amp_rms.to_value(u.nm)
    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    Zs = poppy.zernike.arbitrary_basis(mask, nterms=3, outside=0)
    
    Zc_amp = lstsq(Zs, wfe_amp)
    Zc_opd = lstsq(Zs, wfe_opd)
    for i in range(3):
        wfe_amp -= Zc_amp[i] * Zs[i]
        wfe_opd -= Zc_opd[i] * Zs[i]
    wfe_amp += 1

    wfe = wfe_amp * xp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= poppy.CircularAperture(radius=diam/2).get_transmission(wf)
    
    if plot:
        imshows.imshow2(xp.abs(wfe), xp.angle(wfe)*wavelength.to_value(u.m)/(2*np.pi),
                        npix=npix,
                        vmin1=1-3*amp_rms, vmax1=1+3*amp_rms)

    return wfe

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c

def create_zernike_modes(pupil_mask, nmodes=15, remove_modes=0):
    if remove_modes>0:
        nmodes += remove_modes
    zernikes = poppy.zernike.arbitrary_basis(pupil_mask, nterms=nmodes, outside=0)[remove_modes:]

    return zernikes

def map_acts_to_dm(actuators, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact, Nact))
    command.ravel()[dm_mask.ravel()] = actuators
    return command

# Create control matrix
def WeightedLeastSquares(A, weight_map, nprobes=2, rcond=1e-1):
    control_mask = weight_map > 0
    w = weight_map[control_mask]
    for i in range(nprobes-1):
        w = xp.concatenate((w, weight_map[control_mask]))
    W = xp.diag(w)
    print(W.shape, A.shape)
    cov = A.T.dot(W.dot(A))
    return xp.linalg.inv(cov + rcond * xp.diag(cov).max() * xp.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix also known as the Jacobian
    sts = xp.matmul(S.T, S)
    rho = xp.diag(sts)
    alpha2 = rho.max()

    control_matrix = xp.matmul( xp.linalg.inv( sts + alpha2*10.0**(beta)*xp.eye(sts.shape[0]) ), S.T)
    return control_matrix

def create_circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w//2), int(h//2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = xp.ogrid[:h, :w]
    dist_from_center = xp.sqrt((X - center[0] + 1/2)**2 + (Y - center[1] + 1/2)**2)

    mask = dist_from_center <= radius
    return mask

# Creating focal plane masks
def create_annular_focal_plane_mask(sysi, 
                                    inner_radius, outer_radius, 
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0,
                                    plot=False):
    x = (xp.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2)*sysi.psf_pixelscale_lamD
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r < outer_radius) * (r > inner_radius)
    if edge is not None: mask *= (x > edge)
    
    mask = _scipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
    mask = _scipy.ndimage.shift(mask, (shift[1], shift[0]), order=0)
    
    if plot:
        imshows.imshow1(mask)
        
    return mask

def create_box_focal_plane_mask(sysi, x0, y0, width, height):
    x = (xp.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2)*sysi.psf_pixelscale_lamD
    x,y = xp.meshgrid(x,x)
    x0, y0, width, height = (params['x0'], params['y0'], params['w'], params['h'])
    mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )
    return mask > 0


def masked_rms(image,mask=None):
    return np.sqrt(np.mean(image[mask]**2))

def create_random_probes(rms, alpha, dm_mask, fmin=1, fmax=17, nprobes=3, 
                         plot=False,
                         calc_responses=False):
    # randomized probes generated by PSD
    shape = dm_mask.shape
    ndm = shape[0]

    probes = []
    for n in range(nprobes):
        fx = np.fft.rfftfreq(ndm, d=1.0/ndm)
        fy = np.fft.fftfreq(ndm, d=1.0/ndm)
        fxx, fyy = np.meshgrid(fx, fy)
        fr = np.sqrt(fxx**2 + fyy**2)
        spectrum = ( fr**(alpha/2.0) ).astype(complex)
        spectrum[fr <= fmin] = 0
        spectrum[fr >= fmax] = 0
        cvals = np.random.standard_normal(spectrum.shape) + 1j * np.random.standard_normal(spectrum.shape)
        spectrum *= cvals
        probe = np.fft.irfft2(spectrum)
        probe *= dm_mask * rms / masked_rms(probe, dm_mask)
        probes.append(probe.real)
        
    probes = np.asarray(probes)/rms
    
    if plot:
        for i in range(nprobes):
            if calc_responses:
                response = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift( pad_or_crop(probes[i], 4*ndm) ))))
                imshows.imshow2(probes[i], response, pxscl2=1/4)
            else:
                imshows.imshow1(probes[i])
                
                
    
    return probes

def create_hadamard_modes(dm_mask): 
    Nacts = dm_mask.sum().astype(int)
    np2 = 2**int(xp.ceil(xp.log2(Nacts)))
    hmodes = xp.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = xp.where(dm_mask.flatten().astype(int))
    for hmode in hmodes:
        hmode = hmode[:Nacts]
        mode = xp.zeros((dm_mask.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)
    had_modes = xp.array(had_modes)
    
    return had_modes

def create_fourier_modes(sysi, control_mask, fourier_sampling=0.75, use='both', return_fs=False):
    xfp = (np.linspace(-sysi.npsf/2, sysi.npsf/2-1, sysi.npsf) + 1/2) * sysi.psf_pixelscale_lamD
    fpx, fpy = np.meshgrid(xfp,xfp)
    
    intp = scipy.interpolate.interp2d(xfp, xfp, ensure_np_array(control_mask)) # setup the interpolation function
    
    xpp = np.linspace(-sysi.Nact/2, sysi.Nact/2-1, sysi.Nact) + 1/2
    ppx, ppy = np.meshgrid(xpp,xpp)
    
    fourier_lim = fourier_sampling * int(np.round(xfp.max()/fourier_sampling))
    xfourier = np.arange(-fourier_lim-fourier_sampling/2, fourier_lim+fourier_sampling, fourier_sampling)
    fourier_x, fourier_y = np.meshgrid(xfourier, xfourier) 
    
    # Select the x,y frequencies for the Fourier modes to calibrate the dark hole region
    fourier_grid_mask = ( (intp(xfourier, xfourier) * (((fourier_x!=0) + (fourier_y!=0)) > 0)) > 0 )
    
    fxs = fourier_x.ravel()[fourier_grid_mask.ravel()]
    fys = fourier_y.ravel()[fourier_grid_mask.ravel()]
    sampled_fs = np.vstack((fxs, fys)).T
    
    cos_modes = []
    sin_modes = []
    for f in sampled_fs:
        fx = f[0]/sysi.Nact
        fy = f[1]/sysi.Nact
        cos_modes.append( ( np.cos(2 * np.pi * (fx * ppx + fy * ppy)) * ensure_np_array(sysi.dm_mask) ).flatten() ) 
        sin_modes.append( ( np.sin(2 * np.pi * (fx * ppx + fy * ppy)) * ensure_np_array(sysi.dm_mask) ).flatten() )
    if use=='both' or use=='b':
        modes = cos_modes + sin_modes
    elif use=='cos' or use=='c':
        modes = cos_modes
    elif use=='sin' or use=='s':
        modes = sin_modes
    
    if return_fs:
        return np.array(modes), sampled_fs
    else:
        return np.array(modes)

def create_fourier_probes(sysi, control_mask,
                          fourier_sampling=0.25, 
                          shift=(0,0), 
                          nprobes=2, 
                          plot=False, 
                          calc_responses=False): 
#     make probe modes from the sum of the cos and sin fourier modes
    fourier_modes = create_fourier_modes(sysi, control_mask, fourier_sampling=fourier_sampling, use='both')
    nfs = fourier_modes.shape[0]//2
    Nact = sysi.Nact
    
    probes = np.zeros((nprobes, sysi.Nact, sysi.Nact))
    sum_cos = fourier_modes[:nfs].sum(axis=0).reshape(Nact,Nact)
    sum_sin = fourier_modes[nfs:].sum(axis=0).reshape(Nact,Nact)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    if not isinstance(shift, list):
        shifts = [shift]*nprobes
    else:
        shifts = shift
    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = scipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/np.max(probe)
        
        if plot: 
            response = xp.abs(xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift( pad_or_crop(xp.array(probes[i]), 4*Nact) ))))
            imshows.imshow2(probes[i], response, pxscl2=1/4)
            
    return probes

def fourier_mode(lambdaD_yx, rms=1, acts_per_D_yx=(34,34), Nact=34, phase=0):
    '''
    Allow linear combinations of sin/cos to rotate through the complex space
    * phase = 0 -> pure cos
    * phase = np.pi/4 -> sqrt(2) [cos + sin]
    * phase = np.pi/2 -> pure sin
    etc.
    '''
    idy, idx = np.indices((Nact, Nact)) - (34-1)/2.
    
    #cfactor = np.cos(phase)
    #sfactor = np.sin(phase)
    prefactor = rms * np.sqrt(2)
    arg = 2*np.pi*(lambdaD_yx[0]/acts_per_D_yx[0]*idy + lambdaD_yx[1]/acts_per_D_yx[1]*idx)
    
    return prefactor * np.cos(arg + phase)

def create_probe_poke_modes(Nact, 
                            poke_indices,
                            plot=False):
    Nprobes = len(poke_indices)
    probe_modes = np.zeros((Nprobes, Nact, Nact))
    for i in range(Nprobes):
        probe_modes[i, poke_indices[i][1], poke_indices[i][0]] = 1
    if plot:
        fig,ax = plt.subplots(nrows=1, ncols=Nprobes, dpi=125, figsize=(10,4))
        for i in range(Nprobes):
            im = ax[i].imshow(probe_modes[i], cmap='viridis')
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            fig.colorbar(im, cax=cax)
        plt.close()
        display(fig)
        
    return probe_modes

def create_all_poke_modes(dm_mask):
    Nact = dm_mask.shape[0]
    Nacts = int(np.sum(dm_mask))
    poke_modes = np.zeros((Nacts, Nact, Nact))
    count=0
    for i in range(Nact):
        for j in range(Nact):
            if dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count+=1

    poke_modes = poke_modes[:,:].reshape(Nacts, Nact**2)
    
    return poke_modes

def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '.format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
    
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts
    Xacts,Yacts = np.meshgrid(xacts,yacts)
    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)
    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 
    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
    return probe_commands

def create_sinc_probes(Npairs, Nact, dm_mask, probe_amplitude, probe_radius=10, probe_offset=(0,0), plot=False):
    
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nact, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
            
        probes.append(probe*dm_mask)
    probes = np.array(probes)
    if plot:
        for i,probe in enumerate(probes):
            probe_response = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( pad_or_crop(probe, int(4*Nact))  ))))
            imshows.imshow2(probe, probe_response, pxscl2=1/4)
    
    return probes
    
def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    im = ensure_np_array(im)
    mask = ensure_np_array(mask)
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile
    
def plot_radial_contrast(im, mask, pixelscale, nbins=30, cenyx=None, xlims=None, ylims=None):
    bins, contrast = get_radial_contrast(im, mask, nbins=nbins, cenyx=cenyx)
    r = bins * pixelscale

    fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(6,4))
    ax.semilogy(r,contrast)
    ax.set_xlabel('radial position [$\lambda/D$]')
    ax.set_ylabel('Contrast')
    ax.grid()
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.close()
    display(fig)

def dm_rms(dm_mask, dm_command):
    command = dm_command[dm_mask.ravel]
    
    rms = xp.sqrt(xp.mean(command**2))
    
    return rms
    
def save_fits(fpath, data, header=None, ow=True, quiet=False):
    data = ensure_np_array(data)
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

# functions for saving python objects
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data  


