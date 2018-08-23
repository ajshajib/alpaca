# -*- coding: utf-8 -*-
"""
Module containing several physical constant values.
"""
import numpy as np


class Constants(object):
    """
    Contains the physical constants definitions in cgs units.
    """
    c_kms = 299792.458  # km/s
    c = 29979245800  # cm/s
    LIGHT_SPEED = c

    # Planck constant
    h = 6.62606885e-27  # erg/s
    PLANCK = h
    hbar = h/2./np.pi

    # Stefan-Boltzmann constant
    sigma_sb = 5.670367e-5  # in cgs (erg/cm^2/s/K^4)
    STEFAN_BOLTZMANN = sigma_sb

    # Boltzmann constant
    k_b = 1.38064852e-16  # erg/K
    BOLTZMANN = k_b
