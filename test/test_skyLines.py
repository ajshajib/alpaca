# -*- coding: utf-8 -*-
"""
Tests for `skyLines` module.
"""

import numpy as np

from fabspec.skyLines import SkyLines


class TestSkyLines(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_sky_lines(self):
        """
        Test the methods in `SkyLines` class.
        :return:
        """
        # Check if code runs without error.
        sl = SkyLines()
        sl.setup_wavelengths(start=1900, end=2300)
        sl.get_sky_spectra()
        sl.reset_temperatures(temperature_rot=200, temperature_vib=9500)
        sl.get_sky_spectra()

    @classmethod
    def teardown_class(cls):
        pass
