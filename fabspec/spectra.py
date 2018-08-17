"""
This module defines the class Spectra() that contains a spectra and
relevant information.
"""

import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy


class Spectra(object):
    """
    Contains a spectra and relevant information.
    """

    def __init__(self, spectra, wavelengths, *args, **kwargs):
        """

        :param spectra: Array of flux.
        :param wavelengths: Array of wavelengths, must be same length as
        spectra.
        :param args:
        :param kwargs: `resolution`: R, `resolution_fwhm`: FWHM of
        spectral resolution.
        """
        try:
            assert len(spectra) == len(wavelengths)
        except:
            raise ('Error: spectra and wavelength must have same size!')

        self._spectra = np.array(spectra)  # to store the original spectra
        self.spectra = deepcopy(self._spectra)
        self._wavelengths = np.array(wavelengths) # to store the original
        self.wavelengths = deepcopy(self._wavelengths)

        if 'resolution_fwhm' in kwargs:
            self._resolution_fwhm = kwargs['resolution_fwhm']

        # resolution parameter R
        if 'resolution' in kwargs:
            self._resolution = kwargs['resolution']

        if 'flux_unit' in kwargs:
            self._flux_unit = kwargs['flux_unit']
        else:
            self._flux_unit = 'arbitrary'

        if 'wavelength_unit' in kwargs:
            self._wavelength_unit = kwargs['wavelength_unit']

    @property
    def resolution_fwhm(self):
        if hasattr(self, '_resolution_fwhm'):
            return self._resolution_fwhm
        else:
            return None

    @resolution_fwhm.setter
    def resolution_fwhm(self, fwhm):
        """
        Update the FWHM of the spectra.
        :param fwhm: FWHM to set for the spectra, in the same unit as
        `self.wavelengths`.
        """
        self._resolution_fwhm = fwhm

    @property
    def resolution(self):
        if hasattr(self, '_resolution'):
            return self._resolution
        else:
            return None

    @property
    def flux_unit(self):
        if hasattr(self, '_flux_unit'):
            return self._flux_unit
        else:
            return None

    @property
    def wavelength_unit(self):
        if hasattr(self, '_wavelength_unit'):
            return self._wavelength_unit
        else:
            return None

    def get_delta_lambda(self):
        """
        Compute the spatial pixel size of the spectra.
        :return:
        """
        return np.mean(np.diff(self.wavelengths))

    def linearize_wavelength_scale(self, dlambda):
        """
        Linearize the wavelength scale if its currently in log scale.
        :param dlambda: Wavelength resolution for linear intervals.
        :return:
        """
        sample = interp1d(self.wavelengths, self.spectra, kind='linear')
        # shortening the wavelength range by 1 index so that
        # `scipy.interpolate.interp1d` does not throw error
        self.wavelengths = np.arange(self.wavelengths[1], self.wavelengths[-2],
                                     dlambda)

        self.spectra = sample(self.wavelengths)

    def normalize_flux(self):
        """
        Normalize the flux so that the median is 1.
        :return:
        """
        self.spectra /= np.median(self.spectra)
        self._flux_unit = 'normalized'

    def reset_to_initial(self):
        """
        Reset the spectra to initial flux and wavelengths at the time
        of creating the `Spectra` object.
        :return:
        """
        self.wavelengths = deepcopy(self._wavelengths)
        self.spectra = deepcopy(self._spectra)

    def get_wavelength_range(self):
        """
        Get the wavelength range of the spectra.
        :return:
        """
        return self.wavelengths[[0, -1]] \
               + np.array([-0.5, 0.5])*self.get_delta_lambda()

    def clip(self, start_wavelength, end_wavelength):
        """
        Clip the spectra within the specified wavelengths.
        :param start_wavelength: Start wavelength for clipping.
        :param end_wavelength: End wavelength for clipping.
        :return:
        """
        self.spectra = self.spectra[(self.wavelengths >= start_wavelength) &
                                    (self.wavelengths <= end_wavelength)]
        self.wavelengths = self.wavelengths[(self.wavelengths >=
                                             start_wavelength) &
                                            (self.wavelengths <=
                                             end_wavelength)]