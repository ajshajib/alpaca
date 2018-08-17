"""
Module to simulate a spectra observed by a telescope given the
instruments specifications.
"""

import numpy as np
from scipy.interpolate import interp1d
#from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy

#from .spectra import Spectra

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
        spectra.spectra = np.zeros(len(new_wavelengths))
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


class SimSpec(SimSpecUtil):
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

