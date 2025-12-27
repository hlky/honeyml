import os
from setuptools import find_packages, setup

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "src", "dinoml", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]

setup(
    name="dinoml",
    version=__version__,
    description="DinoML",
    author="hlky",
    author_email="hlky@hlky.ac",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["jinja2", "numpy", "sympy", "click"],
    python_requires=">=3.7, <4",
    zip_safe=True,
)
