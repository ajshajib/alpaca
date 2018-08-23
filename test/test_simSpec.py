"""
Tests for `simSpec` module.
"""

import numpy as np
from pathlib import Path
from astropy.io import fits

from fabspec import Spectra
from fabspec.simSpec import SimSpec
from fabspec.simSpec import SimSpecUtil as sim_util

_PARENT_DIR = Path(__file__).resolve().parents[1]


class TestSimSpecUtil(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_apply_poisson_noise(self):
        """
        Test `SimSpecUtil.apply_poisson_noise()`.
        :return:
        :rtype:
        """
        xs = np.linspace(0, 1., 1000)
        ys = np.ones(len(xs))

        spectra = Spectra(xs, ys)

        sns = np.arange(20, 101, 20)
        for sn in sns:
            spectra = sim_util.apply_poisson_noise(spectra, sn)

            np.testing.assert_allclose(
                np.sqrt(np.mean((spectra.spectra-1.)**2)), 1./sn,
                atol=0.01, rtol=0.01
            )
            spectra.reset_to_initial()

    def test_air_vacuum_wavelength_conversions(self):
        """
        Test the methods in `SkyLines` class.
        :return:
        """
        wavelength = 1529.6  # nm

        # test vacuum to air using Ciddor
        wavelength_air = sim_util.vacuum_to_air_wavelength_ciddor(wavelength)
        np.testing.assert_allclose(wavelength/1.00027328, wavelength_air,
                                   atol=1e-8)

        # test air to vacuum using Ciddor
        wavelength_vac = sim_util.air_to_vacuum_wavelength(wavelength_air,
                                                           formula='ciddor')
        np.testing.assert_allclose(wavelength, wavelength_vac,
                                   atol=1e-8)

        # test vacuum to air using Mathar
        wavelength_air = sim_util.vacuum_to_air_wavelength_mathar(wavelength,
                                                            temperature=288.15)
        np.testing.assert_allclose(wavelength/1.00027332, wavelength_air,
                                   atol=1e-8)

        # test air to vacuum using Mathar
        wavelength_vac = sim_util.air_to_vacuum_wavelength(wavelength_air,
                                                            temperature=288.15)
        np.testing.assert_allclose(wavelength, wavelength_vac,
                                   atol=1e-8)

    @classmethod
    def teardown_class(cls):
        pass


class TestSimSpec(object):

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

        template = Spectra(template_wavelengths, template_spectra)
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