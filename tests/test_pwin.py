import os

import numpy as np
import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.io.espresso.inputs import PWin
from pymatgen.io.espresso.inputs.pwin import (
    AtomicSpeciesCard,
    AtomicPositionsCard,
    KPointsCard,
    AdditionalKPointsCard,
    CellParametersCard,
    ConstraintsCard,
    OccupationsCard,
    AtomicVelocitiesCard,
    AtomicForcesCard,
    SolventsCard,
    HubbardCard,
)

# class AtomicSpeciesCard(InputCard):
# class AtomicPositionsCard(InputCard):
# class KPointsCard(InputCard):
# class AdditionalKPointsCard(InputCard):
# class CellParametersCard(InputCard):
# class ConstraintsCard(InputCard):
# class OccupationsCard(InputCard):
# class AtomicVelocitiesCard(InputCard):
# class AtomicForcesCard(InputCard):
# class SolventsCard(InputCard):
# class HubbardCard(InputCard):


@parametrize_cases(
    Case(
        "kpoints",
        card=KPointsCard,
        args=("crystal_b", [], [], [], [], []),
        kwargs={},
        test_str = ["crystal_b", "alat"]
    ),
    Case(
        "kpoints",
        card=AdditionalKPointsCard,
        args=("crystal_b", [], [], []),
        kwargs={},
        test_str = ["crystal_b", "alat"]
    ),
    Case(
        "cell_parameters",
        card=CellParametersCard,
        args=("angstrom", [1,0,0], [0,1,0], [0,0,1]),
        kwargs={},
        test_str = ["angstrom", "alat"]
    ),
    Case(
        "atomic_positions",
        card=AtomicPositionsCard,
        args=("angstrom", ["H", "H"], [[0,0,0], [0,0,0.7]], None),
        kwargs={},
        test_str = ["angstrom", "alat"]
    )
)
def test_card_options(card, args, kwargs, test_str):
    c = card(*args, **kwargs)
    assert isinstance(c.option, c.opts)
    assert c.option == test_str[0]
    assert c.option != test_str[1]
