from setuptools import setup

setup(name='deppar',
      version='0.0.1',
      author='Tianze Shi, Adrian Benton, Igor Malioutov, Ozan Irsoy',
      author_email='see@the.paper',
      description='Code for reproducing the experiments in:\n\nTianze Shi, Adrian Benton, Igor Malioutov, and Ozan Irsoy. "Diversity-Aware Batch Active Learning for Dependency Parsing". NAACL. 2021.',
      url=' https://github.com/tzshi/dpp-al-parsing-naacl21',
      packages=['deppar'],
      install_requires=['cython',
                        'fire',
                        'numpy',
                        'scipy',
                        'sklearn',
                        'tensorflow',
                        'torch',
                        'transformers==3.0.2'])
