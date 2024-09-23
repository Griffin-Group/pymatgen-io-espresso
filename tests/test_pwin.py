import contextlib
import warnings

import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.core.structure import Structure
from pymatgen.core.units import bohr_to_ang
from pymatgen.io.espresso.inputs import PWin
from pymatgen.io.espresso.inputs.base import EspressoInputWarning
from pymatgen.io.espresso.inputs.pwin import (
    AdditionalKPointsCard,
    AtomicPositionsCard,
    CellParametersCard,
    KPointsCard,
)
from pymatgen.io.espresso.utils import IbravUntestedWarning


@parametrize_cases(
    Case(
        "kpoints",
        card=KPointsCard,
        args=("crystal_b", [], [], [], [], []),
        kwargs={},
        test_str=["crystal_b", "alat"],
    ),
    Case(
        "kpoints",
        card=AdditionalKPointsCard,
        args=("crystal_b", [], [], []),
        kwargs={},
        test_str=["crystal_b", "alat"],
    ),
    Case(
        "cell_parameters",
        card=CellParametersCard,
        args=("angstrom", [1, 0, 0], [0, 1, 0], [0, 0, 1]),
        kwargs={},
        test_str=["angstrom", "alat"],
    ),
    Case(
        "atomic_positions",
        card=AtomicPositionsCard,
        args=("angstrom", ["H", "H"], [[0, 0, 0], [0, 0, 0.7]], None),
        kwargs={},
        test_str=["angstrom", "alat"],
    ),
)
def test_card_options(card, args, kwargs, test_str):
    c = card(*args, **kwargs)
    assert isinstance(c.option, c.opts)
    assert c.option == test_str[0]
    assert c.option != test_str[1]


@parametrize_cases(
    Case(
        "Sr3PbO, ibrav=0",
        mat="Sr3PbO",
        ibrav=0,
        alat=None,
        symbols=["Sr", "Pb", "O"],
        valid=True,
    ),
    Case(
        "Ni, ibrav=2",
        mat="Ni",
        ibrav=2,
        alat=6.648 * bohr_to_ang,
        symbols=["Ni"],
        valid=True,
    ),
    Case(
        "Si, ibrav=2",
        mat="Si",
        ibrav=2,
        alat=10.26 * bohr_to_ang,
        symbols=["Si"],
        valid=True,
    ),
    Case(
        "Bi2Te3, ibrav=0",
        mat="Bi2Te3",
        ibrav=0,
        alat=None,
        symbols=["Bi", "Te"],
        valid=True,
    ),
    Case(
        "BAs, ibrav=0, alat",
        mat="BAs",
        ibrav=0,
        alat=1,
        symbols=["B", "As"],
        valid=True,
    ),
    Case(
        "GaSe, ibrav=0, missing species card",
        mat="GeSe",
        ibrav=0,
        alat=None,
        symbols=["Ge", "Se"],
        valid=False,
    ),
    Case(
        "Al, ibrav=2, missing control namelist",
        mat="Al",
        ibrav=2,
        alat=7.50 * bohr_to_ang,
        symbols=["Al"],
        valid=False,
    ),
)
def test_pwin_structure(mat, ibrav, alat, symbols, valid):
    try:
        pwin = PWin.from_file(f"data/{mat}/bands.in")
    except FileNotFoundError:
        pwin = PWin.from_file(f"data/{mat}/scf.in")
    assert pwin.system["ibrav"] == ibrav
    with pytest.warns(IbravUntestedWarning) if ibrav > 0 else contextlib.nullcontext():
        s1 = pwin.structure
    s2 = Structure.from_file(f"data/{mat}/POSCAR")
    assert s1 == s2
    with contextlib.nullcontext() if alat else pytest.raises(ValueError):
        assert pwin.alat == pytest.approx(alat)
    with warnings.catch_warnings():
        # Deals with more ibrav warnings
        warnings.simplefilter("ignore")
        assert pwin.site_symbols.sort() == symbols.sort()

    # Check if input validator works
    with contextlib.nullcontext() if valid else pytest.warns(EspressoInputWarning):
        assert pwin.validate() == valid
