from setuptools import setup

setup(
    name='RobustSTL',
    version='0.1.1',
    description=('Unofficial Implementation of RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series (AAAI 2019)'),
    author='Aria Ghora Prabono',
    author_email='hello@ghora.net',
    url='https://github.com/ariaghora/RobustSTL',
    license='MIT',
    packages=['rstl'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8'],
    )