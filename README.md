pymatgen-io-espresso
=========================

`pymatgen.io.espresso` is a `pymatgen` addon that adds support for Quantum ESPRESSO. This package has two goals:

1. Elevate Quantum ESPRESSO to a first-class citizen in the `pymatgen`-driven computational materials science ecosystem
2. Provide a public API and classes fully compatible with those from `pymatgen.io.vasp`, so that if your code does this

```python
from pymatgen.io.vasp.outsput import Vasprun

my_dft = Vasprun('vasprun.xml', **kwargs)

# Complicated code
```

It can be rewritten like this:

```python
from pymatgen.io.espresso.pwxml import PWxml

my_dft = PWxml('prefix.xml', **possibly_different_kwargs)

# Exact same complicated code, with no changes
```

The hope is that this will ultimately allow more pacakges that use `pymatgen` under the hood for parsing DFT calculations to support Quantum ESPRESSO with as little developer effort as possible.

Usage
=====

Currently, the package is in pre-alpha and is not available on `PyPi`. To install it:

```bash
git clone https://github.com/oashour/pymatgen-io-espresso.git
cd pymatgen-io-espresso
# RECOMMENDED: activate a virtual environment
pip install .
```

Documentation is currently being worked on.
