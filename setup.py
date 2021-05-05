from setuptools import setup, find_packages

setup(
    name="fsiren",
    version="0.0.1",
    description="Siren neural networks in Flax",
    author="Antonio Stanziola",
    author_email="a.stanziola@ucl.ac.uk",
    packages=["fsiren"],
    install_requires=["flax"]
)