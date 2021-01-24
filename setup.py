import io

from setuptools import find_packages
from setuptools import setup

setup(
    name="flaskr",
    version="1.0.0",
    license="MIT",
    maintainer="Danylo Mocherniuk",
    maintainer_email="danylo1999@gmail.com",
    description="Basic tool to play with ML models.",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["flask"],
    extras_require={"test": ["pytest", "coverage"]},
)