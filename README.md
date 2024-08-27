# pymatgen.io.espresso
[![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-red)](https://oashour.github.io/pymatgen-io-espresso/develop/)
[![GitHub Release](https://img.shields.io/github/v/release/oashour/pymatgen-io-espresso?include_prereleases)](https://github.com/oashour/pymatgen-io-espresso/releases)
[![Tests](https://github.com/oashour/pymatgen-io-espresso/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/oashour/pymatgen-io-espresso/actions)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")
[![Stable Docs](https://img.shields.io/badge/Docs-Stable-blue)](https://oashour.github.io/pymatgen-io-espresso/latest/)
[![Develop Docs](https://img.shields.io/badge/Docs-Develop-purple)](https://oashour.github.io/pymatgen-io-espresso/develop/)

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



# Usage

Currently, the package is in pre-alpha and is not available on `PyPi`. To install it:

```bash
git clone https://github.com/oashour/pymatgen-io-espresso.git
cd pymatgen-io-espresso
pip install .
```

Documentation is currently being worked on.

## Contributing
Contributions are welcome! Please see the [contributing guide](CONTRIBUTING.md) for more information.

## Contributors
* Omar A. Ashour (@oashour): Creator and maintainer.
* Ella Banyas (@ebanyas): Sous-dev. Contributed caffeination module.
