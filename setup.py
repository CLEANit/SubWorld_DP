from setuptools import setup, find_packages

setup(name='subworld-dp',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'cmocean',
          'pyyaml',
          'os',
          'copy',
      ],
      description='Implementation of dynamic programming to solve the POMDP SubWorld.',
      author='Chris Beeler',
      url='https://github.com/CLEANit/SubWorld_DP',
      version='0.0')