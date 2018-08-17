#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://fabspec.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

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
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.6',
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
