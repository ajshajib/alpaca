# -*- coding: utf-8 -*-
"""
Module to simulate a spectra observed by a telescope given the
instruments specifications.
"""

import numpy as np
from scipy.interpolate import interp1d
#from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from copy import deepcopy


class SimSpecUtil(object):
    """
    Contains utility methods to simulate spectra.
    """

    @classmethod
    def oversample(cls, spectra, oversample_ratio=20, conserve_flux=True):
        """
        Oversample spectra by `oversample_ratio`.
        :param spectra: `Spectra` object.
        :param oversample_ratio: Ratio to oversample the wavelength axis.
        :param conserve_flux: If `True` total flux will be conserved.
        :type conserve_flux: bool
        :return:
        """
        if conserve_flux:
            total_flux = np.sum(spectra.spectra) * spectra.get_delta_lambda()

        sample = interp1d(spectra.wavelengths, spectra.spectra, kind='linear',
                          bounds_error=False, fill_value=0)
        spectra.wavelengths = np.linspace(spectra.wavelengths[0],
                                       spectra.wavelengths[-1],
                                       oversample_ratio
                                          * len(spectra.wavelengths))
        spectra.spectra = sample(spectra.wavelengths)

        if conserve_flux:
            spectra.spectra *= total_flux / np.sum(spectra.spectra) \
                               / spectra.get_delta_lambda()

        return spectra

    @classmethod
    def add_velocity_dispersion(cls, spectra, sigma, fwhm=None, verbose=False):
        """
        Add velocity dispersion to the spectra.
        :param spectra: `Spectra` object.
        :param sigma: Velocity dispersion in km/s.
        :param fwhm: Intrinsic fwhm of the spectra.
        :param verbose:
        :return:
        """
        lambda_c = np.mean(spectra.wavelengths)
        dlambda = np.mean(np.diff(spectra.wavelengths))

        if fwhm is None:
            fwhm = spectra.resolution_fwhm
            if fwhm is None:
                fwhm = 0.

        # find out the additional dispersion needed to get final
        # dispersion of sigma
        std_dev_0 = fwhm / 2.355 / dlambda
        std_dev = np.sqrt((sigma / 299792.458 * lambda_c / dlambda) ** 2
                          - std_dev_0 ** 2)

        if verbose:
            print('Standard deviation for velocity dispersion convolution: '
                  '{:.2f} piexels'.format(std_dev))

        spectra.spectra = gaussian_filter1d(spectra.spectra, std_dev)

        return spectra

    @classmethod
    def apply_poisson_noise(cls, spectra, sn, noise_spectra=None):
        """
        Apply Poissonian noise to the spectra.
        :param spectra: `Spectra` object.
        :param sn: Signal-to-noise ratio.
        :param noise_spectra: Noise spectra if not constant noise over
        wavelength.
        :return:
        """
        # Detect if a signed image was input
        # if self.spectra.min() < 0:
        #    low_clip = -1.
        # else:
        #    low_clip = 0.

        # vals = len(np.unique(spectra))
        # vals = 2 ** np.ceil(np.log2(vals))

        # vals = sn**2/np.median(spectra) * np.loadtxt('./real_noise.txt')
        if noise_spectra is None:
            noise_spectra = 1.
        vals = sn ** 2 / np.median(spectra.spectra) / noise_spectra \
               * np.median(noise_spectra)

        # Ensure image is exclusively positive
        # if low_clip == -1.:
        #    old_max = self.spectra.max()
        #    self.spectra = (spectra + 1.) / (old_max + 1.)

        # Generating noise for each unique value in image.
        spectra.spectra = np.random.poisson(spectra.spectra * vals) / vals

        # Return image to original range if input was signed
        # if low_clip == -1.:
        #    self.spectra = self.spectra * (old_max + 1.) - 1.

        return spectra

    @classmethod
    def convolve(cls, spectra, fwhm, verbose=False):
        """
        Convovle the instrumental FWHM into the spectra.
        :param spectra: `Spectra` object.
        :param inst_fwhm: FWHM of the convolution kernel.
        :param verbose:
        :return:
        """
        d_lambda = np.mean(np.diff(spectra.wavelengths))
        std_dev = fwhm / 2.355 / d_lambda  # in pixels

        if verbose:
            print('Standard deviation for instrumental dispersion convolution:'
                  ' {:.2f} piexels.'.format(std_dev))

        spectra.spectra = gaussian_filter1d(spectra.spectra, std_dev)

        return spectra

    @classmethod
    def rebin(cls, spectra, new_wavelengths, res=100, conserve_flux=True):
        """
        Rebin the spectra with given `new_wavelengths`. This converts
        the oversampled spectra to an observed spectra
        by an instrument, for example.
        :param new_wavelengths: Final wavelengths after rebinning.
        :param res: Resolution for integration for each bin.
        :param conserve_flux: If `True` total flux will be conserved.
        :type conserve_flux: bool
        :return:
        """
        if conserve_flux:
            total_flux = np.sum(spectra.spectra) * spectra.get_delta_lambda()

        sample = interp1d(spectra.wavelengths, spectra.spectra, kind='linear',
                          bounds_error=False, fill_value=0)
        spectra.spectra = np.zeros_like(new_wavelengths)
        new_dlambda = np.mean(np.diff(new_wavelengths))

        for i in np.arange(len(new_wavelengths)):
            divs = np.linspace(new_wavelengths[i] - new_dlambda / 2.,
                               new_wavelengths[i] + new_dlambda / 2., res)
            #try:
            integrand = sample(divs)
            #except:
            #    print(new_wavelengths[i] - new_dlambda / 2.,
            #                   new_wavelengths[i] + new_dlambda / 2.)
            #    print(spectra.wavelengths[0], spectra.wavelengths[-1])
            #  raise('')
            spectra.spectra[i] = np.sum(integrand) * new_dlambda / res
            # self.spectra[i] = quad(sample, new_wavelengths[i] -
            #                       new_dlambda / 2.,
            #                       new_wavelengths[i] + new_dlambda / 2.)[0]

        spectra.wavelengths = new_wavelengths

        if conserve_flux:
            spectra.spectra *= total_flux / np.sum(spectra.spectra) \
                               / spectra.get_delta_lambda()

        return spectra

    @classmethod
    def redshift_wavelength(cls, spectra, z, conserve_flux=True):
        """
        Apply a redshift to the wavelength.
        :param spectra: `Spectra` object.
        :param z: Redshift.
        :param conserve_flux: If `True` total flux will be conserved.
        :type conserve_flux: bool
        """
        spectra.wavelengths *= (1.+z)

        if conserve_flux:
            spectra.spectra /= (1.+z)

        return spectra

    @classmethod
    def vacuum_to_air_wavelength_ciddor(cls, wavelength):
        """
        Convert wavelength from vacuum to air. Use the values from Ciddor
        (1996; https://doi.org/10.1364/AO.35.001566). This function is valid
        for 0.3-1.69 micron.
        :param wavelength: Wavelength in vacuum (in nm).
        :return: Wavelength in air (in nm).
        """
        wavelength *= 1e-3  # to micron
        dispersion = 0.05792105 / (
                    238.0185 - wavelength ** -2) + 0.00167917 / (
                             57.362 - wavelength ** -2)

        return wavelength / (1 + dispersion) * 1e3

    @classmethod
    def vacuum_to_air_wavelength_mathar(cls, wavelength, temperature=279.15,
                                        pressure=101325, humidity=0):
        """
        Convert wavelength from vacuum to air. Use the values from Mathar
        (2007; https://doi.org/10.1088/1464-4258/9/5/008), valid for 1.3-2.5
        micron.
        :param wavelength: Wavelength in vacuum (in nm).
        :param temperature: Temperature of air, in K. Default is for 6 degree (
        typical of the upper stratosphere (30-50 km above the ground).
        :param pressure: Pressure in Pascal.
        :param humidity: Humidity in 0-100%.
        :return: Wavelength in air (in nm).
        """
        wavelength *= 1e-3  # to micron.

        # model parameters
        c_ref = [0.200192e-3, 0.113474e-9, -0.424595e-14, 0.100957e-16,
                 -0.293315e-20, 0.307228e-24]  # cm^j
        c_t = [0.588625e-1, -0.385766e-7, 0.888019e-10, -0.567650e-13,
               0.166615e-16, -0.174845e-20]  # cm^j · K
        c_tt = [-3.01513, 0.406167e-3, -0.514544e-6, 0.343161e-9,
                -0.101189e-12, 0.106749e-16]  # cm^j · K^2
        c_h = [-0.103945e-7, 0.136858e-11, -0.171039e-14, 0.112908e-17,
               -0.329925e-21, 0.344747e-25]  # cm^j · %^-1
        c_hh = [0.573256e-12, 0.186367e-16, -0.228150e-19, 0.150947e-22,
                -0.441214e-26, 0.461209e-30]  # cm^j · %^-2
        c_p = [0.267085e-8, 0.135941e-14, 0.135295e-18, 0.818218e-23,
               -0.222957e-26, 0.249964e-30]  # cm^j · Pa^-1
        c_pp = [0.609186e-17, 0.519024e-23, -0.419477e-27, 0.434120e-30,
                -0.122445e-33, 0.134816e-37]  # cm^j · Pa^-2
        c_th = [0.497859e-4, -0.661752e-8, 0.832034e-11, -0.551793e-14,
                0.161899e-17, -0.169901e-21]  # cm^j · K · %^-1
        c_tp = [0.779176e-6, 0.396499e-12, 0.395114e-16, 0.233587e-20,
                -0.636441e-24, 0.716868e-28]  # cm^j · K · Pa^-1
        c_hp = [-0.206567e-15, 0.106141e-20, -0.149982e-23, 0.984046e-27,
                -0.288266e-30, 0.299105e-34]  # cm^j · %^-1 · Pa^-1

        sigma_ref = 1e4 / 2.25  # cm^−1
        temperature_ref = 273.15 + 17.5  # K
        pressure_ref = 75000  # Pa
        humidity_ref = 10  # %

        # model
        sigma = 1e4 / wavelength  # cm^-1
        n = 1  # refractive index
        for j in range(0, 6):
            n += (c_ref[j] + c_t[j] * (1 / temperature - 1 / temperature_ref)
                  + c_tt[j] * (1 / temperature - 1 / temperature_ref) ** 2
                  + c_h[j] * (humidity - humidity_ref)
                  + c_hh[j] * (humidity - humidity_ref) ** 2
                  + c_p[j] * (pressure - pressure_ref)
                  + c_pp[j] * (pressure - pressure_ref) ** 2
                  + c_th[j] * (1 / temperature - 1 / temperature_ref)
                  * (humidity - humidity_ref)
                  + c_tp[j] * (1 / temperature - 1 / temperature_ref)
                  * (pressure - pressure_ref)
                  + c_hp[j] * (humidity - humidity_ref)
                  * (pressure - pressure_ref)) * (sigma - sigma_ref) ** j

        return wavelength / n * 1e3

    @classmethod
    def air_to_vacuum_wavelength(cls, wavelength, formula='mathar',
                                 temperature=279.15, **kwargs):
        """
        Convert wavelength from air to vacuum. Solve the vacuum-to-air
        conversion formula from Ciddor (1996) or Mathar (2007).
        :param wavelength: Wavelength in air (in nm).
        :param formula: Formula to use, options: 'mathar', 'ciddor'.
        :param temperature: Temperature (if not default) for
        `formula='mathar'`.
        :param kwargs: Keyword arguments to pass to `scipy.optimize.brentq`.
        :return: Wavelength in vacuum (in nm).
        """
        assert formula in ['mathar', 'ciddor'], "Invalid formula! Must be " \
                                                "either 'mathar' or 'ciddor'."

        wavelength *= 1e-3  # to micron

        if formula is 'ciddor':
            dispersion = wavelength * 1e3 / cls.vacuum_to_air_wavelength_ciddor(
                wavelength * 1e3) - 1.
        else:
            dispersion = wavelength * 1e3 / cls.vacuum_to_air_wavelength_mathar(
                wavelength * 1e3, temperature=temperature) - 1.

        def func(x):
            if formula is 'ciddor':
                return (cls.vacuum_to_air_wavelength_ciddor(x * 1e3)
                        - wavelength * 1e3)
            else:
                return (cls.vacuum_to_air_wavelength_mathar(x * 1e3,
                                                            temperature=temperature)
                        - wavelength * 1e3)

        high = (dispersion * 2 + 1.) * wavelength
        low = wavelength

        assert func(low) <= 0., 'The function to solve is not negative ' \
                                'at lower limit! {} {}'.format(func(low), low)

        while func(high) <= 0.:
            high += wavelength * dispersion * 2.

        if 'xtol' not in kwargs:
            kwargs['xtol'] = 1e-10

        return brentq(func, low, high, **kwargs) * 1e3


class SimSpec(object):
    """
    Class to simulate spectra observed by an instrument
    """
    def __init__(self, template, wavelength_range, delta_lambda,
                 resolution=None, fwhm_instrument=None, redshift=0):
        """
        :param template: `Spectra` object.
        :param wavelength_range: Tuple, wavelength range of the instrument.
        :param resoltuion: Resolution, R = \lambda/FWHM.
        :param delta_lambda: Size of spatial pixel.
        :param instrumental_fwhm: Instrumental resolution in FWHM.
        :param signal_to_noise: Signal-to-noise ratio.
        :param redshift: Redshift.
        """
        assert resolution is not None or fwhm_instrument is not None

        self._template = template
        self._wavelength_range = wavelength_range
        self._resolution = resolution
        self._delta_lambda = delta_lambda
        if fwhm_instrument is None:
            self._fwhm_instrument = (wavelength_range[1]-wavelength_range[0])\
                                    / 2 / self._resolution
        else:
            self._fwhm_instrument = fwhm_instrument
        self._redshift = redshift

    def simulate(self, vel_dis, signal_to_noise, noise_spectra=None,
                 oversample_ratio=20., res=100, verbose=False):
        """
        Simulate the spectra.
        :param vel_dis: Velocity dispersion to add, in km/s.
        :param signal_to_noise: S/N for Poissonian noise. -1 for none.
        :param noise_spectra: Noise spectra if variable noise level.
        :param oversample_ratio: Oversample ratio.
        :param res: Resolution for rebinning integral.
        :param verbose:
        :return:
        """
        sim_util = SimSpecUtil()

        spectra = deepcopy(self._template)
        spectra = sim_util.redshift_wavelength(spectra, self._redshift)
        spectra = sim_util.oversample(spectra,
                                      oversample_ratio=oversample_ratio)
        spectra = sim_util.add_velocity_dispersion(spectra, sigma=vel_dis,
                                                fwhm=spectra.resolution_fwhm,
                                                verbose=verbose)
        spectra = sim_util.convolve(spectra, fwhm=self._fwhm_instrument,
                                    verbose=verbose)
        instrument_wavelengths = np.arange(self._wavelength_range[0]
                                             + self._delta_lambda/2.,
                                             self._wavelength_range[1],
                                             self._delta_lambda)
        spectra = sim_util.rebin(spectra, instrument_wavelengths, res=res)

        if signal_to_noise > 0.:
            #spectra.normalize_flux()
            spectra = sim_util.apply_poisson_noise(spectra, sn=signal_to_noise,
                                               noise_spectra=noise_spectra)

        return spectra

