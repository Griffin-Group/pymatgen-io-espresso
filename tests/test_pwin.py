
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.io.espresso.inputs.pwin import (
    AdditionalKPointsCard,
    AtomicPositionsCard,
    CellParametersCard,
    KPointsCard,
)


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
