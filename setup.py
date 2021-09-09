import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dagmc_h5m_file_inspector",
    version="develop",
    author="The dagmc_h5m_file_inspector Development Team",
    author_email="mail@jshimwell.com",
    description="Extracts information from DAGMC h5m files including volumes number, material tags",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fusion-energy/dagmc_h5m_file_inspector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=[
        "pytest-cov",
        "pytest-runner",
        "requests"
    ],
    install_requires=[
        # pymoab is required by is not available on pypi, however moab is on conda
        # "moab",
    ],
)
