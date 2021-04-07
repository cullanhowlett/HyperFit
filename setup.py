import pathlib
from setuptools import setup, find_packages
from src.hyperfit import __version__

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="hyperfit",
    version=__version__,
    description="Properly fit data with x and y errors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CullanHowlett/HyperFit",
    author="Cullan Howlett",
    author_email="cullan.howlett@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
    ],
    package_dir={"": "src"},
    packages=["hyperfit"],
    python_requires=">=3.9, <4",
    py_modules=["os", "inspect", "abc"],
    install_requires=["numpy>=1.20.0", "scipy>=1.6.0", "zeus-mcmc>=2.3.0", "pandas>=1.2.0", "emcee>=3.0.0"],
    package_data={"hyperfit": ["data/*.txt"]},
    project_urls={
        "Bug Reports": "https://github.com/CullanHowlett/HyperFit/issues",
    },
)
