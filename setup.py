from __future__ import print_function, unicode_literals, division

# Using setuptools allows the --develop option.
from setuptools import setup

setup(
    name='sheet_manager',
    version='0.1',
    packages=['', 'sheet_manager', 'sheet_manager.gui', 'sheet_manager.omr', 'sheet_manager.omr.utils',
              'sheet_manager.omr.config', 'sheet_manager.omr.models'],
    url='',
    license='(c) All rights reserved.',
    author='Matthias Dorfer',
    author_email='matthias.dorfer@jku.at',
    description='Tool for creating aligned MIDI, audio, and sheet music image data.'
)
