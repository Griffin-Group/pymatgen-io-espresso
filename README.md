# pymatgen.io.espresso
[![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-red)](https://Griffin-Group.github.io/pymatgen-io-espresso/develop/)
[![GitHub Release](https://img.shields.io/github/v/release/Griffin-Group/pymatgen-io-espresso?include_prereleases)](https://github.com/Griffin-Group/pymatgen-io-espresso/releases)
[![Tests](https://github.com/Griffin-Group/pymatgen-io-espresso/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/Griffin-Group/pymatgen-io-espresso/actions)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")
[![Stable Docs](https://img.shields.io/badge/Docs-Stable-blue)](https://Griffin-Group.github.io/pymatgen-io-espresso/latest/)
[![Develop Docs](https://img.shields.io/badge/Docs-Develop-purple)](https://Griffin-Group.github.io/pymatgen-io-espresso/develop/)

`pymatgen.io.espresso` is a `pymatgen` addon that adds support for Quantum ESPRESSO. This package has two goals:

1. Elevate Quantum ESPRESSO to a first-class citizen in the `pymatgen`-driven computational materials science ecosystem
2. Provide a public API and classes fully compatible with those from `pymatgen.io.vasp`, so that if your code does this

```python
from pymatgen.io.vasp.outputs import Vasprun

my_calc = Vasprun('vasprun.xml', **kwargs)

# Complicated code
```

It can be rewritten like this:

```python
from pymatgen.io.espresso.outputs import PWxml

my_calc = PWxml('prefix.xml', **possibly_different_kwargs)

# Exact same complicated code, with no changes
```

The hope is that this will ultimately allow more pacakges that use `pymatgen` under the hood for parsing DFT calculations to support Quantum ESPRESSO with as little developer effort as possible.

The package is currently in pre-alpha testing.

# Usage

Currently, the package is in pre-alpha and is not available on `PyPi`. To install it:

```bash
pip install pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
```

## Contributing
Contributions are welcome! Please see the [contributing guide](CONTRIBUTING.md) for more information.

## Contributors
* Omar A. Ashour (@oashour): Creator and maintainer.
* Ella Banyas (@ebanyas): Sous-dev, bug fixes, caffeination module (WIP).
* Willis O'Leary (@wolearyc): Sous-dev, bug fixes.
