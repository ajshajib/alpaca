"""
Module to simulate OH sky lines spectra.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .constants import Constants as cs
from .spectra import Spectra
from .simSpec import SimSpecUtil as spec_util

_PARENT_DIR = Path(__file__).resolve().parents[1]


class SkyLines(object):
    """
    Contains everything related to OH sky lines.
    """

    MAX_V = 13  #fa maximum vibrational quantum level

    def __init__(self):
        """
        Initiate `SkyLines`. Load data containing line wavelengths, transition
        levels, Einstein's A, etc. The data is from Brooke et al. (2016).
        """
        self.temperature_rot = 190.
        self.temperature_vib = 9000.

        self._wavelength_start = 0.
        self._wavelength_end = 0.
        self._wavelengths = []

        self._qn_vs = pd.DataFrame(columns=['v', 'Q', 'N'],
                                   index=np.arange(self.MAX_V+1))

        self.line_list = pd.DataFrame()  # stores line data
        self._line_list_v = []  # list of lines (dataframes) indexed with v
        self._lines_in_range = pd.DataFrame()  # lines in wavelength range

        self.load_data()  # loads data into `self.line_list`

    def load_data(self):
        """
        Load necessary data files.
        """
        data_file_path = _PARENT_DIR / 'data' / 'OH-XX-Line_list.txt'
        self.line_list = pd.read_csv(data_file_path,
                                     delim_whitespace=True, skiprows=33)

        for v in range(self.MAX_V+1):
            # subset of line_list with v' = v
            v_lines = self.line_list[
                self.line_list["v'"].values == v
                ][["J'", "F'", "p'", "E''", 'Calculated']].drop_duplicates(
            ).reset_index()
            self._line_list_v.append(v_lines)

    def reset_temperatures(self, temperature_rot=190, temperature_vib=9000):
        """
        Reset the temperatures without reloading data.
        :param temperature_rot: Rotational temperature.
        :param temperature_vib: Vibrational temperature.
        """
        self.temperature_rot = temperature_rot
        self.temperature_vib = temperature_vib

        # Resetting all the values in the dataframe, but keeping the same
        # dataframe object for efficient memory allocation.
        self._qn_vs.loc[:] = np.nan
        self._lines_in_range.loc[:, ('Q_v', 'N_v', 'Intensity')] = np.nan

    def setup_wavelengths(self, start=None, end=None, resolution=0.01,
                          wavelengths=None):
        """
        :param start: Beginning wavelength in nm.
        :param end: Ending wavelength in nm.
        :param resolution: Resolution element in nm.
        :param wavelengths: Wavelengths for computing the sky spectra. If
        provided, `start` and `end` are not needed.
        """
        if wavelengths is None:
            assert start is not None and end is not None, 'You need to ' \
                                                          'provide either ' \
                                                          'the wavelengths ' \
                                                          'or the range.'
            self._wavelength_start = start
            self._wavelength_end = end
            self._wavelengths = np.arange(start, end+resolution/2., resolution)
        else:
            self._wavelengths = wavelengths
            self._wavelength_start = wavelengths[0]
            self._wavelength_end = wavelengths[-1]

        k_start = 1e7 / spec_util.air_to_vacuum_wavelength(
            self._wavelength_end)  # in /cm
        k_end = 1e7 / spec_util.air_to_vacuum_wavelength(
            self._wavelength_start)  # in /cm

        self._lines_in_range = self.line_list[
            self.line_list["Calculated"].between(k_start, k_end)]
        self._lines_in_range = self._lines_in_range.assign(
            Q_v=np.nan,
            N_v=np.nan,
            Intensity=np.nan)

    def get_sky_spectra(self, temperature_rot=None, temperature_vib=None):
        """
        Generate the sky spectra between start and end wavelengths.
        :param temperature_rot: Rotational temperature, in K.
        :param temperature_vib: Vibrational temperature, in K.
        """
        if temperature_rot is not None and temperature_vib is not None:
            self.reset_temperatures(temperature_rot=temperature_rot,
                                    temperature_vib=temperature_vib)
        elif temperature_rot is not None:
            self.reset_temperatures(temperature_rot=temperature_rot,
                                    temperature_vib=self.temperature_vib)
        elif temperature_vib is not None:
            self.reset_temperatures(temperature_rot=self.temperature_rot,
                                    temperature_vib=temperature_vib)

        flux = np.zeros_like(self._wavelengths)

        intensities = self._get_line_intensities_in_range()
        line_wavelengths = spec_util.vacuum_to_air_wavelength_mathar(
            1e7/self._lines_in_range['Calculated'].values)  # nm

        pixel_size = self._wavelengths[1] - self._wavelengths[0]
        for line_wavelength, intensity in zip(line_wavelengths, intensities):
            idx = int((line_wavelength - self._wavelength_start)/pixel_size)
            if 0 <= idx < len(flux):
                flux[idx] += intensity

        return Spectra(self._wavelengths, flux, wavelength_unit='nm')

    def _get_qn_v(self, v):
        """
        Compute partition function Q_v (T_rot) and N_v (T_vib). Q_v (T_rot) is
        computed using equation (39) from Mies (1974). Simply, N_v (T_vib) =
        Q_v (T_vib).
        :param v: Vibrational quantum number.
        :type v: int
        """
        if v in self._qn_vs.v.values:
            idx = np.where(self._qn_vs.v.values == v)[0]
            return (self._qn_vs.Q.values[idx],
                    self._qn_vs.N.values[idx])
        else:
            v_lines = self._line_list_v[v]

            wave_n = v_lines["E''"].values + v_lines['Calculated'].values
            j = v_lines["J'"].values

            q = np.sum((4*j+2) * np.exp(-self._get_beta(wave_n,
                                                        self.temperature_rot)
                                        ))
            n = np.sum((4*j+2) * np.exp(-self._get_beta(wave_n,
                                                        self.temperature_vib)
                                        ))

            self._qn_vs.loc[v] = [v, q, n]

            return q, n

    @staticmethod
    def _get_beta(wavenumber, temperature):
        """
        Compute $\beta(E)$ from wavenumber k (cm^-1).
        :param wavenumber: Wavenumber in cm^-1.
        :param temperature: Temperature in K.
        """
        return cs.h * cs.c * wavenumber / cs.k_b / temperature

    @staticmethod
    def _wavenumber2energy(wavenumber):
        """
        Convert wavenumber to energy.
        :param wavenumber: Wavenumber in cm^-1.
        :return:
        """
        return cs.h * cs.c * wavenumber

    def _populate_qn_in_line_list(self):
        """
        Fills up [Q_v, N_v] columns of  `SkyLines._lines_in_range` with
        calculated values.
        :return:
        :rtype:
        """
        for v in range(self.MAX_V+1):
            q_v, n_v = self._get_qn_v(v)
            row_slice = self._lines_in_range["v'"] == v
            self._lines_in_range.loc[row_slice,
                                     'Q_v'] = q_v
            self._lines_in_range.loc[row_slice,
                                     'N_v'] = n_v

    def _get_line_intensities_in_range(self):
        """
        Compute line intensities for lines in `self._lines_in_range`.
        Computed intensity is unit of ergs/cm^3/s. Equation (41) from Mies
        (1974) is used to derive the intensities in photon number, then the
        energy of the line is multiplied to get the intensities in unit of
        energy.
        :param index:
        """
        if np.isnan(self._lines_in_range['Intensity'].values).any():
            self._populate_qn_in_line_list()

            v = self._lines_in_range["v'"].values
            wavenumber = self._lines_in_range['Calculated'].values
            einstein_a = self._lines_in_range['A'].values
            j = self._lines_in_range["J'"].values
            energy_lower = self._lines_in_range["E''"].values
            q_v = self._lines_in_range['Q_v'].values
            n_v = self._lines_in_range['N_v'].values

            intensity = self._wavenumber2energy(wavenumber) * n_v \
                        * einstein_a * (4*j+2) / q_v * np.exp(
                -self._get_beta(energy_lower+wavenumber, self.temperature_rot))

            self._lines_in_range.loc[:, 'Intensity'] = pd.Series(intensity,
                                            index=self._lines_in_range.index)

            return intensity
        else:
            return self._lines_in_range['Intensity'].values
