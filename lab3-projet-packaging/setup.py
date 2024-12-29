from setuptools import setup, find_packages

setup(
    name="LinearRegression",  # Replace with your package name
    version="0.1.0",  # Initial version
    description="This package is for linear regression.",
    author="MBIA NDI Marie Thérèse",
    author_email="mbialaura12@gmail.com",
    packages=find_packages(),  # Automatically find and include all packages
    # install_requires=[
    #     "numpy>=1.21.0",  # List your dependencies here
    #     "scikit-learn>=1.0.0",
    #     "pandas =^2.2.3"
    # ],
    python_requires=">=3.12",  # Minimum Python version
)