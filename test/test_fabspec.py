# -*- coding: utf-8 -*-
"""
Tests for `fabspec` module.
"""

import numpy as np

from fabspec import Spectra


class TestFabspec(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_spectra(self):
        """
        Test the methods in `Spectra` class.
        :return:
        """

        xs = np.arange(0., 100., 1.)
        ys = np.sin(xs)

        spectra = Spectra(xs, ys, flux_unit='test', wavelength_unit='test')

        assert spectra.flux_unit == 'test' \
               and spectra.wavelength_unit == 'test'

        assert (spectra.get_wavelength_range() == np.array([0., 99.])).all()

        spectra.clip(start_wavelength=50., end_wavelength=90.)
        assert (spectra.get_wavelength_range() == np.array([50., 90.])).all()

        spectra.reset_to_initial()
        assert (spectra.get_wavelength_range() == np.array([0., 99.])).all()

        assert spectra.get_delta_lambda() == 1.

        # check if linearization works correctly
        xs = np.logspace(0, 1, 100)
        ys = np.tanh(xs)

        spectra = Spectra(xs, ys, flux_unit='test', wavelength_unit='test')
        spectra.linearize_wavelength_scale(dlambda=1.)
        assert spectra.get_delta_lambda() == 1.
        assert (spectra.get_wavelength_range() == np.array([1., 10.])).all()
        np.testing.assert_allclose(np.tanh(spectra.wavelengths),
                                   spectra.spectra, atol=1e-6,
                                   rtol=1e-4)

    @classmethod
    def teardown_class(cls):
        pass
