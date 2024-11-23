# pymatgen.io.espresso
[![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-red)](https://Griffin-Group.github.io/pymatgen-io-espresso/develop/)
[![GitHub Release](https://img.shields.io/github/v/release/Griffin-Group/pymatgen-io-espresso?include_prereleases)](https://github.com/Griffin-Group/pymatgen-io-espresso/releases)
[![Tests](https://github.com/Griffin-Group/pymatgen-io-espresso/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/Griffin-Group/pymatgen-io-espresso/actions)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")
[![Stable Docs](https://img.shields.io/badge/Docs-Stable-blue)](https://Griffin-Group.github.io/pymatgen-io-espresso/latest/)
[![Develop Docs](https://img.shields.io/badge/Docs-Develop-purple)](https://Griffin-Group.github.io/pymatgen-io-espresso/develop/)

`pymatgen.io.espresso` is a `pymatgen` addon that adds support for Quantum ESPRESSO (QE). The goal of this package is to elevate QE to a first-class citizen in the `pymatgen`-driven computational materials science ecosystem. 

`pymatgen.io.espresso` aims to provide classes whose public APIs are fully compatible with those from `pymatgen.io.vasp`. The intention is to allow any `pymatgen`-based VASP post-processing code to add QE support with little to no developer effort. Ideally, this should be as simple as going from this

```python
from pymatgen.io.vasp.outputs import Vasprun

calc = Vasprun('vasprun.xml', **kwargs)
# Complicated code
```

to this
```python
from pymatgen.io.espresso.outputs import PWxml

calc = PWxml('prefix.xml', **possibly_different_kwargs)
# Exact same complicated code, with no changes
```
without any additional changes to the actual post-processing code. Under the hood, `pymatgen.io.espresso` automatically converts all the units, coordinates, and conventions used by QE to those used by VASP.

`pymatgen.io.espresso` additionally provides utilities for parsing and creating Quantum ESPRESSO input files, and a converter from VASP inputs (`INCAR`, `KPOINTS`, and `POSCAR`) to `pw.x` inputs ("Caffeinator") is a work in progress.

# Usage

Currently, the package is in pre-alpha and is not yet available on `PyPi`. To install it:

```bash
pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
```

We have detailed documentation automatically generated from our doc strings, and some simple tutorials and examples are a work in progress.

## Contributing
Contributions are welcome! Please see the [contributing guide](CONTRIBUTING.md) for more information.

## Contributors
* Omar A. Ashour (@oashour): Creator and maintainer.
* Ella Banyas (@ebanyas): Sous-dev, bug fixes, caffeination module (WIP).
* Willis O'Leary (@wolearyc): Sous-dev, bug fixes.
