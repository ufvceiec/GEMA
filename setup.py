import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="GEMA",
	version = "0.1",
    author="UFV CEIEC",
    author_email="ceiec.info@ceiec.es",
    description="A library to build and study Self-Organizing-Maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ufvceiec/GEMA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)