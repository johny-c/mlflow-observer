from setuptools import setup

from mlflow_observer import __version__

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Utilities",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "License :: OSI Approved :: MIT License",
]


with open("README.rst", "r", encoding="utf-8") as fp:
    long_description = fp.read()

setup(name="mlflow-observer",
      version=__version__,
      author="John Chiotellis",
      author_email="johnyc.code@gmail.com",
      url="https://github.com/johny-c/mlflow-observer",
      install_requires=['sacred', 'mlflow'],
      tests_require=["pytest"],
      py_modules=["mlflow_observer"],
      description="Experiment tracking with sacred and mlflow",
      long_description=long_description,
      license="MIT",
      classifiers=classifiers,
      python_requires=">=3.6",
      )
