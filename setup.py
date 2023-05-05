from setuptools import setup, find_packages

setup(
    name="anndata_dissector",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
