from setuptools import setup
from setuptools import find_packages

setup(name='dlgo',
      version='0.1',
      install_requires=['tensorflow', 'six'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
