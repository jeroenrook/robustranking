import setuptools

setuptools.setup(
    name="robustranking",
    author="Jeroen Rook",
    version="0.0.1",
    packages=setuptools.find_packages(include=["robustranking"]),
    install_requires=[
        "pandas",
        "numpy",
    ]
)