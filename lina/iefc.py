from .math_module import xp, _scipy, ensure_np_array
import lina.utils as utils
from lina.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, pca_modes=None, plot=False):
    N_probes = len(probe_cube)
    
    diff_ims = []
    ims = []
    for i in range(N_probes):
        probe = probe_cube[i]

        sysi.add_dm(probe_amplitude * probe) # add positive probe
        im_pos = sysi.snap()
        sysi.add_dm(-probe_amplitude*probe) # remove positive probe
        sysi.add_dm(-probe_amplitude * probe) # add negative probe
        im_neg = sysi.snap()
        sysi.add_dm(probe_amplitude*probe) # remove negative probe

        diff_ims.append((im_pos - im_neg) / (2*probe_amplitude))

    diff_ims = xp.array(diff_ims)
    # if pca_modes is not None:
    #     differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
    
    if plot:
        for i, diff_im in enumerate(diff_ims):
            imshow2(probe_cube[i], diff_im.reshape(sysi.npsf, sysi.npsf), 
                    f'Probe Command {i+1}', 'Difference Image', pxscl2=sysi.psf_pixelscale_lamD,
                    cmap1='viridis')
    
    return diff_ims
    
def calibrate(sysi, 
              control_mask, 
              probe_amplitude, probe_modes, 
              calibration_amplitude, calibration_modes, 
              scale_factors=None, 
              return_all=False,
              plot_responses=False, 
              ):
    print('Calibrating iEFC...')
    
    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]

    response_matrix = []
    calib_amps = []
    if return_all: # be ready to store the full focal plane responses (difference images)
        response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(sysi.Nact, sysi.Nact)

            if scale_factors is not None: 
                calib_amp = calibration_amplitude * scale_factors[ci]
            else:
                calib_amp = calibration_amplitude

            # Add the mode to the DMs
            sysi.add_dm(s * calib_amp * dm_mode)
            
            # Compute reponse with difference images of probes
            diff_ims = take_measurement(sysi, probe_modes, probe_amplitude)
            calib_amps.append(calib_amp)
            response += s * diff_ims.reshape(Nprobes, sysi.npsf**2) / (2 * calib_amp)
            
            # Remove the mode form the DMs
            sysi.add_dm(-s * calib_amp * dm_mode) # remove the mode
        
        print(f"\tCalibrated mode {ci+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        if probe_modes.shape[0]==2:
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()],
                                                    response[1, control_mask.ravel()]]) )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()], 
                                                    response[1, control_mask.ravel()],
                                                    response[2, control_mask.ravel()]]) )
        
        if return_all: 
            response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    if return_all:
        response_cube = xp.array(response_cube)
    
    if plot_responses:
        dm_response_map = xp.sqrt(xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(sysi.Nact,sysi.Nact) / xp.max(dm_response_map)
        imshow1(dm_response_map, 'DM RMS Actuator Responses', lognorm=True, vmin=1e-2)
            
    if return_all:
        return response_matrix, xp.array(response_cube)
    else:
        return response_matrix
    
def run(sysi,
        control_matrix,
        probe_modes, probe_amplitude, 
        calibration_modes,
        control_mask,
        num_iterations=3,
        loop_gain=0.5, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        plot_probes=False,
        plot_radial_contrast=False,
        all_ims=None, 
        all_commands=None,
       ):
    
    print('Running iEFC...')
    start = time.time()
    starting_itr = len(all_ims)

    Nmodes = calibration_modes.shape[0]
    modal_matrix = calibration_modes.reshape(Nmodes, -1)

    total_coeff = 0.0
    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((sysi.Nact,sysi.Nact))
    for i in range(num_iterations):
        print(f"\tClosed-loop iteration {i+1+starting_itr} / {num_iterations+starting_itr}")
        sysi.subtract_dark = False
        diff_ims = take_measurement(sysi, probe_modes, probe_amplitude, plot=plot_probes)
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        print(modal_matrix.shape, modal_coeff.shape)
        # total_coeff = (1.0-leakage)*total_coeff + loop_gain*modal_coeff
        # total_command = calibration_modes.T.dot(total_coeff).reshape(sysi.Nact,sysi.Nact)
        del_command = modal_matrix.T.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        total_command = (1.0-leakage)*total_command + loop_gain*del_command
        sysi.set_dm(total_command)

        sysi.subtract_dark = True
        image_ni = sysi.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        all_ims.append(copy.copy(image_ni))
        all_commands.append(copy.copy(total_command))
    
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            imshow3(del_command, total_command, image_ni, 
                    f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                    'Total DM Command', 
                    f'Image\nMean NI = {mean_ni:.3e}',
                    cmap1='viridis', cmap2='viridis', 
                    pxscl3=sysi.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)
            
            if plot_radial_contrast:
                utils.plot_radial_contrast(image_ni, control_mask, sysi.psf_pixelscale_lamD, nbins=50,
#                                            ylims=[1e-10, 1e-4],
                                          )
    
    print('Closed loop for given control matrix completed in {:.3f}s.'.format(time.time()-start))
    return all_ims, all_commands

def run_iteration(I,
                control_matrix,
                probe_modes, probe_amplitude, 
                calibration_modes,
                # modal_matrix,
                control_mask,
                gain=1/2,
                leakage=0.0,
                plot=True,
                plot_radial_contrast=False,
                plot_probes=False,
                clear=True,
                all_ims=None, 
                all_commands=None,
                ):
    '''
    
    '''
    I.return_ni = True

    I.subtract_dark = False
    diff_ims = take_measurement(I, probe_modes, probe_amplitude, plot=plot_probes)
    measurement_vector = diff_ims[:, control_mask].ravel()

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = -control_matrix.dot(measurement_vector)
    del_command = calibration_modes.T.dot(modal_coeff).reshape(I.Nact,I.Nact)
    # del_command = modal_matrix.dot(modal_coeff).reshape(I.Nact,I.Nact)

    # maybe we want to implement leakage as just removing a fraction of the previous
    # command and not just removing a fraction of the total iEFC command
    total_command = I.get_dm()
    total_command = (1.0-leakage)*total_command + gain*del_command
    # total_command = I.get_dm()
    # total_command = total_command - leakage*all_commands[-1] + gain*del_command
    I.set_dm(total_command)

    I.subtract_dark = True
    image_ni = I.snap()
    mean_ni = xp.mean(image_ni[control_mask])

    if all_ims is not None: all_ims.append(copy.copy(image_ni))
    if all_commands is not None: all_commands.append(copy.copy(total_command))

    if plot:
        imshow3(del_command, total_command, image_ni, 
                f'Iteration {1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)
        
        if plot_radial_contrast:
                utils.plot_radial_contrast(image_ni, control_mask, I.psf_pixelscale_lamD, nbins=50,
#                                            ylims=[1e-10, 1e-4],
                                          )
        if clear: clear_output(wait=True)




