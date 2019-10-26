import os
import sys
from setuptools import setup

install_requires = [
    'numpy>=1.14.0',
    'scipy>=1.1.0',
    'scikit-learn',
    'future',
    'torch',
    'graphtools',
    'nmslib'
]

test_requires = [
    'nose',
    'nose2',
]

if sys.version_info[0] == 3:
    test_requires += ['anndata']

doc_requires = [
    'sphinx',
    'sphinxcontrib-napoleon']

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >=3.5 required.")

version_py = os.path.join(os.path.dirname(
    __file__), 'SR3', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

readme = open('README.rst').read()

setup(name='SR3',
      version=version,
      description='SR3 fusion clustering',
      author='Jay S. Stanley III, Gal Mishne, Eric C. Chi, and Ronald R. Coifman',
      author_email='jay.s.stanley.3@gmail.com',
      packages=['SR3', 'SR3.math'],
      license='GNU General Public License Version 2',
      install_requires=install_requires,
      extras_require={'test': test_requires,
                      'docs': doc_requires},
      test_suite='nose2.collector.collector',
      long_description=readme,
      url='https://github.com/stanleyjs/SR3',
      keywords=['tensors',
                'big-data',
                'signal processing',
                'manifold-learning',
                'clustering'
                ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Mathematics',
      ]
      )