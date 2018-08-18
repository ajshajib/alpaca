#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


readme = open('README.rst').read()
doclink = """
Documentation
-------------
The full documentation can be generated with Sphinx"""

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

desc = open("README.rst").read()
requires = ['numpy>=1.13', 'scipy>=0.14.0', "configparser"]
tests_require=['pytest>=2.3', "mock"]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='fabspec',
    version='0.0.1',
    description='Extract stellar or gas kinematics from galaxy '
                'absorption-line IFU or long-slit spectra.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Anowar J. Shajib',
    author_email='ajshajib@gmail.com',
    url='https://github.com/ajshajib/fabspec',
    packages=[
        'fabspec',
    ],
    package_dir={'fabspec': 'fabspec'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='fabspec',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
        #'Programming Language :: Python :: Implementation :: PyPy',
    ],
    tests_require=tests_require,
    cmdclass={'test': PyTest},#'build_ext':build_ext,
)
