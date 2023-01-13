import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quandelight",  # Replace with your own username
    version="0.0",
    author="Lou Bernabeu",
    author_email="bernabeu.lou@gmail.com",
    description="A Python package to simulate Quandela's cavities and optimize their properties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MechanicalPenguin225/quandelight",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
