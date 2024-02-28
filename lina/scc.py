from .math_module import xp, _scipy
from . import utils
from . import imshows
import time
import copy

from IPython.display import display, clear_output

def estimate_coherent(sysi, r_npix=0, shift=(0,0), image=None, dark_mask=None, plot=False):
    '''
    r_npix:
        radius of sidebands in units of pixels
    shift:
        location of sideband centers in pixels (from center of array)
    '''
    if image is None:
        im = sysi.snap()
    else:
        im = image
    
    if dark_mask is not None:
        im *= dark_mask

    im_max = im.max()
    
    im_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(im), norm='ortho'))
    # im_fft_sum = xp.sum(xp.abs(im_fft))
    
    if plot:
        imshows.imshow2(xp.abs(im_fft), xp.angle(im_fft), lognorm1=True)
    im_fft_shift = _scipy.ndimage.shift(im_fft, shift)
    
    x = xp.linspace(-im.shape[0]//2, im.shape[0]//2-1, im.shape[0]) + 1/2
    x,y = xp.meshgrid(x,x)
    
    r = xp.sqrt(x**2 + y**2)
    mask = r<r_npix
    im_fft_masked = mask*im_fft_shift
    
    # im_fft_masked_sum = xp.sum(xp.abs(im_fft_masked))
    # im_fft_masked *= xp.sqrt((im_fft_sum-im_fft_masked_sum)/im_fft_masked_sum)
    
    if plot:
        imshows.imshow3(mask, xp.abs(im_fft_shift), xp.abs(im_fft_masked), lognorm2=True, lognorm3=True)
    
    E_est = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(im_fft_masked), norm='ortho'))

    if dark_mask is not None:
        E_est *= dark_mask

    norm = (xp.abs(E_est) ** 2).max()
    E_est *= xp.sqrt(im_max / norm)

    return E_est

def estimate_incoherent():
    '''
    FIXME
    '''
    
    return E_est

def build_jacobian(sysi, 
                   epsilon, 
                   control_mask,
                   control_modes,
                   imnorm,
                   plot=False,
                   **scc_kwargs,
                   ):
    '''
    This can be done on the actual testbed with individual pokes
    '''
    start = time.time()
    
    amps = xp.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    # if hasattr(sysi, 'bad_acts'):
    #     dm_mask[sysi.bad_acts] = False
    
    Nacts = int(dm_mask.sum())
    Nmask = int(control_mask.sum())
    
    num_modes = control_modes.shape[0]
    modes = control_modes 
    
    responses = xp.zeros((2*Nmask, num_modes))
    count = 0

    print('Calculating Jacobian: ')
    for i in range(num_modes):
        response = 0
        for amp in amps:
            mode = modes[i].reshape(sysi.Nact,sysi.Nact)

            sysi.add_dm(utils.ensure_np_array(amp.get() * mode))
            wavefront = estimate_coherent(sysi, dark_mask=None, **scc_kwargs) / xp.sqrt(imnorm)
            response += amp * wavefront.ravel() / (2*xp.var(amps))
            sysi.add_dm(utils.ensure_np_array(-amp.get() * mode))
        
        responses[::2,count] = response[control_mask.ravel()].real
        responses[1::2,count] = response[control_mask.ravel()].imag

        if plot:
            imshows.imshow1(xp.abs(response.reshape(256, 256)) ** 2, lognorm=True)
            time.sleep(2)
            clear_output(wait=True)
        
        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count+1, num_modes, 
                                                                                            time.time()-start), end='')
        print("\r", end="")
        count += 1
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return responses

def run(sysi, 
        control_matrix,
        control_mask, 
        control_modes,
        imnorm,
        jacobian=None,
        gain=0.5, 
        iterations=5, 
        plot_all=False, 
        plot_current=True,
        plot_radial_contrast=True,
        **scc_kwargs,
        ):
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    if jacobian is not None:
        _, s, _ = xp.linalg.svd(jacobian, full_matrices=False)
        alpha2 = xp.max( xp.diag( xp.real( jacobian.conj().T @ jacobian ) ) )
        print('Max singular value squared:\t', s.max()**2)
        print('alpha^2:\t\t\t', alpha2) 
    
    Nmask = int(control_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()
    # if hasattr(sysi, 'bad_acts'):
    #     dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm()
    dm_command = utils.ensure_np_array(xp.zeros((sysi.Nact, sysi.Nact))) 
    efield_ri = xp.zeros(2*Nmask)

    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations), end="\r")
        sysi.set_dm(dm_ref + dm_command)
        E_est = estimate_coherent(sysi, dark_mask=None, **scc_kwargs) / xp.sqrt(imnorm)
        I_est = xp.abs(E_est)**2
        I_exact = sysi.snap() / imnorm

        # rms_est = xp.sqrt(xp.mean(I_est[control_mask]**2))
        # rms_im = xp.sqrt(xp.mean(I_exact[control_mask]**2))
        # mf = rms_est/rms_im # measure how well the estimate and image match

        commands.append(sysi.get_dm())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))

        efield_ri[::2] = E_est.ravel()[control_mask.ravel()].real
        efield_ri[1::2] = E_est.ravel()[control_mask.ravel()].imag

        # del_dm = -control_matrix.dot(efield_ri)
        # del_dm = sysi.map_actuators_to_command(del_dm)
        # dm_command += gain * del_dm

        del_modes = -control_matrix.dot(efield_ri)

        del_dm = utils.ensure_np_array(del_modes).dot(control_modes)
        del_dm = xp.array(del_dm).reshape(sysi.Nact, sysi.Nact)
        dm_command += gain * utils.ensure_np_array(del_dm)

        if plot_current or plot_all:
            if not plot_all: clear_output(wait=True)

            imshows.imshow3(commands[i], I_est, I_exact, 
                            lognorm2=True, lognorm3=True)

            if plot_radial_contrast:
                utils.plot_radial_contrast(images[-1], control_mask, sysi.psf_pixelscale_lamD, nbins=100)

        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images