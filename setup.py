from setuptools import setup, find_packages

setup(
    name = 'internalTools',
    version= '0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines()
)