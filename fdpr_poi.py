from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.propagation import focus_fixed_sampling
from prysm.geometry import circle,spider
"""
from prysm.polynomials.zernike import (
    zernike_nm,
    noll_to_nm,
    zernike_nm_seq
)
from prysm.polynomials.__init__ import (
    hopkins,
    sum_of_2d_modes
)
"""
from prysm.polynomials import (
    noll_to_nm,
    zernike_nm,
    zernike_nm_seq,
    hopkins,
    sum_of_2d_modes
)
from prysm import mathops, conf
mathops.set_backend_to_cupy()
# conf.config.precision = 32

from prysm.mathops import (np,
                           fft,
                           interpolate,)

import matplotlib.pyplot as plt

import sys
from lina.phase_retrieval import ADPhaseRetireval, ParallelADPhaseRetrieval

from scipy.optimize import minimize


# parameters
D_pupil = 6500                              # mm
D_obs = 1300                                # mm
D_spider = 150                              # mm
fno = 15                                    # dimensionless
efl = fno * D_pupil                         # mm
wvls = list(np.linspace(0.575, 0.725, 25).get())  # um [0.633]
npix_pupil = 512                            # pix
dx_pupil = D_pupil / npix_pupil


# grids
x, y = make_xy_grid(npix_pupil, diameter=D_pupil)
r, t = cart_to_polar(x, y)
extent_pupil = [-D_pupil / 2, D_pupil / 2, -D_pupil / 2, D_pupil / 2]

# aperture
s1 = spider(1, D_spider, x, y, rotation=0)
s2 = spider(1, D_spider, x, y, rotation=120)
s3 = spider(1, D_spider, x, y, rotation=240)
spiders = s1 & s2 & s3
A = (circle(D_pupil / 2, r) ^ circle(D_obs / 2, r)) & spiders
# A = circle(D_pupil / 2, r)

plt.imshow(A.get(), extent=extent_pupil ,cmap='gray')
plt.title("6.5-m Aperture")
plt.xlabel("mm")

# zernike basis
r_norm = r / (D_pupil / 2)
nms = [noll_to_nm(i) for i in range(2, 37)]
zernikes = list(zernike_nm_seq(nms, r, t, norm=True))
zernikes = [z / np.max(np.abs(z)) for z in zernikes]

# random phase error to estimate
np.random.seed(20240820)
zernike_coeffs = np.random.random(len(nms)) * 0.2
opd = sum_of_2d_modes(zernikes, zernike_coeffs)

plt.imshow(opd.get() * A.get(), extent=extent_pupil, vmin=-0.75, vmax=0.75, cmap='coolwarm')
plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
plt.title(f"RMS WFE: {np.sqrt(np.mean(opd[opd != 0] ** 2)):0.2f} um")
plt.xlabel("mm")


def fwd(opd, defocus_values):
    # defocus OPDs
    # Defocus OPD from what to what using what?
    #   hopkins(a,b,c,r,t,H) = abs(sin(a)*t) * r**b * H**c ; if a<0
    #   hopkins(a,b,c,r,t,H) = cos(a)*t * r**b * H**c ; if a>=0
    #   r_norm are radial coordinates normalized to the radius of the telescope primary
    #   t is the azimuthal radial coordinate (t == theta)

    #   H020 is the defocus term

    defocus_opds = [hopkins(0, 2, 0, r_norm, t, 0) * val for val in defocus_values]

    psf_list = []

    # propagate PSFs
    for defocus_opd in defocus_opds:
        psf = 0
        for wvl in wvls:
            k = 2 * np.pi / wvl  # Define wavenumbers
            wf = A * np.exp(1j * k * opd)  # Define wavefront (A is pupil) => WF = Pupil * exp(ik(OPD))
            #   OPD looks like just a product of Zernike modes and coefficients. Dimensionless.
            #   If OPD and defocus OPD have the same dimensions, then the exponential arguments in the two expressions
            #   wf = A exp(2*i*pi/lambda * OPD)
            #   defocus_wf = A exp(2*i*pi*defocusOPD)
            #   These are dimensionally inconsistent?
            #   Also, when we plug this into the wavefunction, we get A*A * exp(), should we have A**2 here?
            #   It will be fine if it is a binary mask I guess, but is it generally a bad idea?

            defocus = A * np.exp(-2j * np.pi * defocus_opd)  # Defocussed OPD is converted to

            mono = focus_fixed_sampling(wavefunction=wf * defocus,
                                        input_dx=dx_pupil,
                                        prop_dist=efl,
                                        wavelength=wvl,
                                        output_dx=3.76,
                                        output_samples=128)
            psf += np.abs(mono) ** 2 / len(wvls)

        psf_list.append(psf)

    return psf_list


defocus_values = np.asarray([0, 0.25, 0.5, 0.75, 1])

psfs = fwd(opd, defocus_values)

for i, psf in enumerate(psfs):
    plt.figure(figsize=(5, 4))
    plt.imshow(psf.get(), cmap='magma', norm='log', vmin=1e-1)
    plt.colorbar(pad=0.04, fraction=0.046)
    plt.title(f"Defocus: {defocus_values[i]:0.2f}$\lambda$")

adpr_list = []

for defocus_value, psf in zip(defocus_values, psfs):
    adpr_list.append(ADPhaseRetireval(amp=A,
                                          amp_dx=dx_pupil,
                                          efl=efl,
                                          wvls=wvls,
                                          basis=np.asarray(zernikes),
                                          target=psf,
                                          img_dx=3.76,
                                          defocus_waves=defocus_value,
                                          initial_phase=None))

fdpr = ParallelADPhaseRetrieval(optlist=adpr_list)

results = minimize(fdpr.fg, x0=np.zeros(len(zernike_coeffs)).get(),
                   jac=True, method='L-BFGS-B',
                   options={'maxls': 20, 'ftol': 1e-20, 'gtol': 1e-8, 'disp': 1, 'maxiter':1000})

for i, opt in enumerate(fdpr.optlist):
    plt.plot(np.asarray(opt.cost).get(), label=f'Defocus: {defocus_values[i]:0.2f}$\lambda$', alpha=0.4)
plt.ylabel('MSE')
plt.xlabel('Iterations')
plt.legend(loc='upper right')
plt.yscale('log')
plt.show()

adpr_focus = fdpr.optlist[1]
plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(opd.get() * A.get(), extent=extent_pupil, vmin=-0.75, vmax=0.75, cmap='coolwarm')
plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
plt.title(f"True RMS WFE: {np.sqrt(np.mean(opd[opd != 0] ** 2)):0.3f} um")
plt.xlabel("mm"); plt.yticks([])
plt.subplot(132)
plt.imshow(adpr_focus.phs.get() * A.get(), extent=extent_pupil, vmin=-0.75, vmax=0.75, cmap='coolwarm')
plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
plt.title(f"Estimate RMS WFE: {np.sqrt(np.mean(adpr_focus.phs[adpr_focus.phs != 0] ** 2)):0.3f} um")
plt.xlabel("mm"); plt.yticks([])
diff = opd - adpr_focus.phs
plt.subplot(133)
plt.imshow(diff.get() * A.get(), extent=extent_pupil, vmin=-0.001, vmax=0.001, cmap='coolwarm')
plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
plt.title(f"RMS Estimate Error: {np.sqrt(np.mean(diff[diff != 0] ** 2)):0.5f} um")
plt.xlabel("mm"); plt.yticks([])

for adpr in fdpr.optlist:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title('Target')
    plt.imshow(adpr.D.get(), norm='log', cmap='magma', vmin=1e-1)
    plt.colorbar()
    plt.subplot(122)
    plt.title('Estimate')
    plt.imshow(adpr.I.get(), norm='log', cmap='magma', vmin=1e-1)
    plt.colorbar()
    plt.show()

# all the zernikes
r_norm = r / (D_pupil / 2)
nms = [noll_to_nm(i) for i in range(2, 37)]
zernikes = list(zernike_nm_sequence(nms, r, t, norm=True))
zernikes = [z / np.max(np.abs(z)) for z in zernikes]

# field dependent zernikes
nms_field = [noll_to_nm(i) for i in range(2, 9)] # tip/tilt, defocus, astig, coma
zernikes_field = list(zernike_nm_sequence(nms_field, r, t, norm=True))
zernikes_field = [z / np.max(np.abs(z)) for z in zernikes_field]

# common zernikes
nms_common = [noll_to_nm(i) for i in range(9, 50)] # the rest
zernikes_common = list(zernike_nm_sequence(nms_common, r, t, norm=True))
zernikes_common = [z / np.max(np.abs(z)) for z in zernikes_common]

# common phase error
np.random.seed(69)
zernike_coeffs = np.random.random(len(zernikes)) * 0.2
opd = sum_of_2d_modes(zernikes, zernike_coeffs)

# field dependent phase errors
opds = []
opds_field = []
np.random.seed(420)
for i in range(len(psfs)):
    zernike_field_coeffs = np.random.uniform(-1, 1, len(nms_field)) * 0.05
    opds_field.append(sum_of_2d_modes(zernikes_field, zernike_field_coeffs))
    opds.append(opd + opds_field[-1])

for i, (opd_field, opd_total) in enumerate(zip(opds_field, opds)):

    plt.figure(figsize=(17, 5))
    plt.subplot(131)
    plt.imshow(opd.get() * A.get(), extent=extent_pupil, vmin=-0.75, vmax=0.75, cmap='coolwarm')
    plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
    plt.title(f"Common\nRMS: {np.sqrt(np.mean(opd[A.astype(bool)] ** 2)):0.4f} um")
    plt.xlabel("mm"); plt.yticks([])

    plt.subplot(132)
    plt.imshow(opd_field.get() * A.get(), extent=extent_pupil, vmin=-0.1, vmax=0.1, cmap='coolwarm')
    plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
    plt.title(f"Field-Dependent\nRMS: {np.sqrt(np.mean(opd_field[A.astype(bool)] ** 2)):0.4f} um")
    plt.xlabel("mm"); plt.yticks([])

    plt.subplot(133)
    plt.imshow(opd_total.get() * A.get(), extent=extent_pupil, vmin=-0.75, vmax=0.75, cmap='coolwarm')
    plt.colorbar(pad=0.04, fraction=0.046, label="OPD (um)")
    plt.title(f"Total\nRMS: {np.sqrt(np.mean(opd_total[A.astype(bool)] ** 2)):0.4f} um")
    plt.xlabel("mm"); plt.yticks([])

    plt.suptitle(f"Field Position {i + 1:0.0f}, {defocus_values[i]:0.2f} Waves of Defocus")


def fwd(opds, defocus_values):
    # defocus OPDs
    defocus_opds = [hopkins(0, 2, 0, r_norm, t, 0) * val for val in defocus_values]

    psf_list = []

    # propagate PSFs
    for i, opd in enumerate(opds):
        psf = 0
        for wvl in wvls:
            k = 2 * np.pi / wvl
            wf = A * np.exp(1j * k * opd)
            defocus = A * np.exp(-2j * np.pi * defocus_opds[i])
            mono = focus_fixed_sampling(wavefunction=wf * defocus,
                                        input_dx=dx_pupil,
                                        prop_dist=efl,
                                        wavelength=wvl,
                                        output_dx=3.76,
                                        output_samples=128)
            psf += np.abs(mono) ** 2 / len(wvls)

        psf_list.append(psf)

    return psf_list


defocus_values = np.asarray([0, 0.25, 0.5, 0.75, 1])

psfs = fwd(opds, defocus_values)

for i, psf in enumerate(psfs):
    plt.figure(figsize=(5, 4))
    plt.imshow(psf.get(), cmap='magma', norm='log', vmin=1e-1)
    plt.colorbar(pad=0.04, fraction=0.046)
    plt.title(f"Defocus: {defocus_values[i]:0.2f}$\lambda$")


def bing_bong(psfs, parameters, disp=True):
    results = []

    for i, parameter in enumerate(parameters):
        if parameter['type'] == 'joint':

            # get relevant info from parameters
            ind_modes = parameter['independent_modes']
            common_modes = parameter['common_modes']
            ind_gain = parameter['independent_gain']
            common_gain = parameter['common_gain']
            options = parameter['options']

            # construct modes and gains for joint FDPR
            modes = np.asarray(ind_modes + common_modes)
            gains = np.concatenate((np.ones((len(ind_modes),)) * ind_gain, np.ones((len(common_modes),)) * common_gain))

            # get initial phase estimates for joint FDPR
            if i == 0:
                initial_phases = []
                for j in range(len(psfs)):
                    initial_phases.append(None)
            else:
                initial_phases = results[-1]['opd_estimates']

            # make list of ADPR classes
            adpr_list = []
            for j in range(len(psfs)):
                adpr_list.append(ADPhaseRetireval(amp=A,
                                                  amp_dx=dx_pupil,
                                                  efl=efl,
                                                  wvls=wvls,
                                                  basis=modes * gains[:, None, None],
                                                  target=psfs[j],
                                                  img_dx=3.76,
                                                  defocus_waves=defocus_values[j],
                                                  initial_phase=initial_phases[j]))

            # shove em into FDPR and optimize
            fdpr = ParallelADPhaseRetrieval(optlist=adpr_list)
            result = minimize(fdpr.fg, x0=np.zeros(len(modes)).get(), jac=True, method='L-BFGS-B', options=options).x

            # get the optimized modal coefficients
            coeffs = []
            if i == 0:
                for j in range(len(psfs)):
                    coeffs.append(result * gains.get())
            else:
                for j in range(len(psfs)):
                    coeffs.append(results[-1]['coefficients'][j] + result * gains.get())

            # turn those coefficients into some OPD maps
            opd_estimates = []
            for j in range(len(psfs)):
                opd_estimates.append(sum_of_2d_modes(modes, coeffs[j]))

            # grab the PSF estimates and error function costs as well
            psf_estimates = []
            costs = []
            for opt in fdpr.optlist:
                psf_estimates.append(opt.I.get())
                costs.append(np.asarray(opt.cost).get())

            # throw in results
            results.append({'coefficients': coeffs,
                            'opd_estimates': opd_estimates,
                            'psf_estimates': psf_estimates,
                            'costs': costs})

            # if desired, display costs
            if disp:
                for j, opt in enumerate(fdpr.optlist):
                    plt.plot(np.asarray(opt.cost).get(), label=f'Defocus : {defocus_values[j]:0.2f}$\lambda$',
                             alpha=0.4)
                plt.title('BING BONG ITERATION ' + str(i + 1))
                plt.ylabel('MSE')
                plt.xlabel('Iterations')
                plt.legend(loc='upper right')
                plt.yscale('log')
                plt.show()

        elif parameter['type'] == 'individual':

            # get relevant info from parameters
            ind_modes = parameter['independent_modes']
            common_modes = parameter['common_modes']
            ind_gain = parameter['independent_gain']
            common_gain = parameter['common_gain']
            options = parameter['options']

            # construct modes and gains for individual ADPR routines
            modes = np.asarray(ind_modes + common_modes)
            gains = np.concatenate((np.ones((len(ind_modes),)) * ind_gain, np.ones((len(common_modes),)) * common_gain))

            # get initial phase estimates for individual ADPR routines
            if i == 0:
                initial_phases = []
                for j in range(len(psfs)):
                    initial_phases.append(None)
            else:
                initial_phases = results[-1]['opd_estimates']

            # make list of ADPR classes
            adpr_list = []
            for j in range(len(psfs)):
                adpr_list.append(ADPhaseRetireval(amp=A,
                                                  amp_dx=dx_pupil,
                                                  efl=efl,
                                                  wvls=wvls,
                                                  basis=modes * gains[:, None, None],
                                                  target=psfs[j],
                                                  img_dx=3.76,
                                                  defocus_waves=defocus_values[j],
                                                  initial_phase=initial_phases[j]))

            # optimize individual ADPR routines and get the estimated coefficients
            coeffs = []
            for j in range(len(adpr_list)):
                result = minimize(adpr_list[j].fg, x0=np.zeros(len(modes)).get(), method='L-BFGS-B', jac=True,
                                  options=options).x
                if i == 0:
                    coeffs.append(result * gains.get())
                else:
                    coeffs.append(results[-1]['coefficients'][j] + result * gains.get())

            # turn those coefficients into some OPD maps
            opd_estimates = []
            for j in range(len(psfs)):
                opd_estimates.append(sum_of_2d_modes(modes, coeffs[j]))

            # grab the PSF estimates and error function costs as well
            psf_estimates = []
            costs = []
            for opt in fdpr.optlist:
                psf_estimates.append(opt.I.get())
                costs.append(np.asarray(opt.cost).get())

            # throw in results
            results.append({'coefficients': coeffs,
                            'opd_estimates': opd_estimates,
                            'psf_estimates': psf_estimates,
                            'costs': costs})

            # if desired, display costs
            if disp:
                for j, opt in enumerate(adpr_list):
                    plt.plot(np.asarray(opt.cost).get(), label=f'Defocus : {defocus_values[j]:0.2f}$\lambda$',
                             alpha=0.4)
                plt.title('BING BONG ITERATION ' + str(i + 1))
                plt.ylabel('MSE')
                plt.xlabel('Iterations')
                plt.legend(loc='upper right')
                plt.yscale('log')
                plt.show()

    return results

bing_bong_parameters = [{'type'              : 'joint',
                         'independent_modes' : zernikes_field,
                         'common_modes'      : zernikes_common,
                         'independent_gain'  : 0.9,
                         'common_gain'       : 1,
                         'options'           : {'maxls' : 20, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}},
                        {'type'              : 'individual',
                         'independent_modes' : zernikes_field,
                         'common_modes'      : zernikes_common,
                         'independent_gain'  : 1,
                         'common_gain'       : 0.9,
                         'options'           : {'maxls' : 20, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 200}},
                        {'type'              : 'joint',
                         'independent_modes' : zernikes_field,
                         'common_modes'      : zernikes_common,
                         'independent_gain'  : 0.5,
                         'common_gain'       : 1,
                         'options'           : {'maxls' : 20, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 200}},
                        {'type'              : 'individual',
                         'independent_modes' : zernikes_field,
                         'common_modes'      : zernikes_common,
                         'independent_gain'  : 1,
                         'common_gain'       : 0.5,
                         'options'           : {'maxls' : 20, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 200}}]

results = bing_bong(psfs, bing_bong_parameters, disp=True)