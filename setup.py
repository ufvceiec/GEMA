import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GEMA",
    version="0.4.3",
    author="UFV CEIEC",
    author_email="ceiec.info@ceiec.es",
    description="A library to build and study Self-Organizing-Maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ufvceiec/GEMA",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "tqdm",
        "pandas",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
