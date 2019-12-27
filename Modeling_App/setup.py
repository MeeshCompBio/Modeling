from setuptools import setup, find_packages

__author__ = "meeshcompbio"
__copyright__ = "Copyright (c) 2019 --, %s" % __author__
__credits__ = ["meeshcompbio"]
__email__ = ""
__maintainer__ = "meeshcompbio"

long_description = 'Just a simple app I made to train a model and be able to return predicitons using FastAPI'

setup(
    name='Modeling_App',
    version="0.0.1",
    packages=find_packages(exclude=[]),
    url='',
    author=__author__,
    author_email=__email__,
    description='',
    long_description=long_description,
    keywords='',
    install_requires=[],
    include_package_data=True,
    extras_require={
    },
)