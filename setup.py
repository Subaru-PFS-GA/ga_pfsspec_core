import os
import re
from io import open
from typing import Any, Match, cast

import pkg_resources
from setuptools import find_packages, setup

# Change the PACKAGE_NAME only to change folder and different name
PFSSPEC_SUBMODULE = 'core'
PACKAGE_ROOT = 'python'
PACKAGE_DIR = 'python/pfs/ga/'
PACKAGE_NAME = 'pfsspec-{}'.format(PFSSPEC_SUBMODULE)
PACKAGE_PPRINT_NAME = "PFSSPEC {}".format(PFSSPEC_SUBMODULE[0].upper() + PFSSPEC_SUBMODULE[1:])
PACKAGE_GITHUB_URL = "https://github.com/Subaru-PFS-GA/ga_pfsspec_{}".format(PFSSPEC_SUBMODULE)

# a-b-c => a/b/c
PACKAGE_FOLDER_PATH = PACKAGE_DIR + '/' + PACKAGE_NAME.replace("-", "/")
# a-b-c => pfs.ga.a.b.c
NAMESPACE_NAME = 'pfs.ga.' + PACKAGE_NAME.replace("-", ".")

# Version extraction inspired from 'requests'
with open(os.path.join(PACKAGE_FOLDER_PATH, "_version.py"), "r") as fd:
    version = cast(Match[Any], re.search(r'^VERSION\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE)).group(1)
if not version:
    raise RuntimeError("Cannot find version information")

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

packages = find_packages(
    where=PACKAGE_ROOT,
    exclude=[
        "test",
        # Exclude packages that will be covered by PEP420 or nspkg
        "pfs",
        "pfs.ga",
        "pfs.ga.pfsspec",
    ]
)

setup(
    package_dir={"": PACKAGE_ROOT},
    name=PACKAGE_NAME,
    version=version,
    description="{} Library for Python".format(PACKAGE_PPRINT_NAME),
    long_description_content_type="text/markdown",
    long_description=readme,
    license="NO LICENSE, DO NOT DISTRIBUTE",
    author="Laszlo Dobos",
    author_email="dobos@jhu.edu",
    url="https://github.com/Subaru-PFS-GA",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: Other/Proprietary License",
    ],
    zip_safe=False,
    include_package_data=True,
    packages=packages,
    python_requires="<4.0,>=3.7",
    install_requires=[
        # NOTE: To avoid breaking changes in a major version bump, all dependencies should pin an upper bound if possible.
        "tqdm<5.0.0",
    ],
    project_urls={
        "Bug Reports": PACKAGE_GITHUB_URL + "/issues",
        "Source": PACKAGE_GITHUB_URL,
    },
)