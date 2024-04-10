from setuptools import setup, find_packages

setup(name='subworlddp',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'cmocean',
          'pyyaml',
      ],
      description='Implementation of dynamic programming with uncertainty to solve the semi-controlled sensing POMDP SubWorld.',
      author='Chris Beeler',
      url='https://github.com/CLEANit/SubWorld_DP',
      version='1.0')
