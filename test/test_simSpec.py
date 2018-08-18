"""
Tests for `simSpec` module.
"""

import numpy as np
from pathlib import Path
from astropy.io import fits

from fabspec import Spectra
from fabspec.simSpec import SimSpec

_PARENT_DIR = Path(__file__).resolve().parents[1]


class TestFabspec(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_simspec(self):
        """
        Test the `SimSpec` class.
        :return:
        """
        z = 0.4
        lambda_start = 0.97 * 1e4  # AA
        lambda_end = 1.42 * 1e4  # AA

        R = 800  # resolution lambda/fwhm
        pixels = 978

        template_file = _PARENT_DIR / 'data'  \
                    / 'xshooter_spec_fits_HD170820_480116_55410_UVB+VIS.fits'

        hdu = fits.open(template_file)
        data = hdu[0].data
        header = hdu[0].header

        template_spectra = data  # /np.median(data)
        template_wavelengths = np.arange(header['NAXIS1']) * header['CDELT1']\
                              + \
                           header['CRVAL1']
        template_wavelengths = np.e ** template_wavelengths

        template = Spectra(template_spectra, template_wavelengths)
        template.clip(start_wavelength=lambda_start/(1. + z)/1.2,
                      end_wavelength=lambda_end/(1. + z)*1.2)

        sim = SimSpec(template, wavelength_range=(lambda_start, lambda_end),
                      delta_lambda=(lambda_end - lambda_start) / pixels,
                      resolution=R, redshift=z)

        simulated_spectra = sim.simulate(vel_dis=200, signal_to_noise=0.)
        simulated_spectra.normalize_flux()

        standard_spectra = np.genfromtxt(_PARENT_DIR / 'data' / 'sim_spec.txt',
                                         delimiter=',')
        standard_spectra[:, 0] /= np.median(standard_spectra[:, 0])

        # Invert the spectra, because for absorption lines, troughs will have
        # higher chance of mismatch.
        np.testing.assert_allclose(1./standard_spectra[:, 0],
                                   1./simulated_spectra.spectra,
                                   rtol=5e-2, atol=1e-2)

        np.testing.assert_allclose(standard_spectra[:, 1],
                                   simulated_spectra.wavelengths, atol=1e-6,
                                   rtol=1e-10)

    @classmethod
    def teardown_class(cls):
        pass
