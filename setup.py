from setuptools import setup, find_packages

setup(
    name="eve_training",
    version="0.1",
    author="Lennar Karstensen",
    packages=find_packages(),
    install_requires=["optuna"],
)
