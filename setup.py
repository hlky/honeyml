import os
from setuptools import find_packages, setup

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "src", "honey", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]

setup(
    name="honey",
    version=__version__,
    description="Honey: Optimization never tasted so sweet 🍯🐝",
    author="hlky",
    author_email="hlky@hlky.ac",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["jinja2", "numpy", "sympy"],
    python_requires=">=3.7, <4",
    zip_safe=True,
)
