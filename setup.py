#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "matplotlib",
    "pandas",
    "seaborn",
    "scikit-learn",
    "joblib",
    "scipy",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3", "scanpy", "python-igraph", "louvain", "pytest-mpl"]

setup(
    author="Maxim Zaslavsky",
    author_email="maxim@maximz.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="General genetics/genomics utilities.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="genetools",
    name="genetools",
    packages=find_packages(include=["genetools", "genetools.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/maximz/genetools",
    version="0.7.2",
    zip_safe=False,
    extras_require={
        # This will also install anndata and UMAP packages
        "scanpy": ["scanpy"]
    },
)
