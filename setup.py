import sys

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

# To use a consistent encoding
from codecs import open
from os import path



__version__ = '0.0.1'

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


if __name__ == '__main__':
    here = path.abspath(path.dirname(__file__))

    # Get the long description from the README file
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()


    setup(
        name='mlcv-tutorial',
        version=__version__,
        description='Assisting library for the ML4CV tutorial',
        long_description=long_description,
        url='https://github.com/johny-c/mlcv-tutorial.git',
        author='John Chiotellis',
        author_email='johnyc.code@gmail.com',
        license='GPLv3',

        classifiers=[
                    'Development Status :: 4 - Beta',
                    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                    'Natural Language :: English',
                    'Operating System :: MacOS :: MacOS X',
                    'Operating System :: Microsoft :: Windows',
                    'Operating System :: POSIX :: Linux',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.5',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Artificial Intelligence'],

        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        package_dir={'mlcv-tutorial': 'mlcv'},
        install_requires=['numpy>=1.13',
                          'scipy>=0.19',
                          'scikit_learn>=0.19',
                          'requests>=2.20'
                          'matplotlib>=2.0'],

        cmdclass = {'test': PyTest},
        test_suite='mlcv.tests.test_mlcv',
        tests_require=['pytest'],
        extras_require={'testing': ['pytest']}
    )
