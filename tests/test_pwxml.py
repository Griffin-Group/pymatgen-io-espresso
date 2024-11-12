"""Tests for PWXml class."""

import numpy as np
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.core.units import Ry_to_eV
from pymatgen.io.espresso.outputs import PWxml


@parametrize_cases(
    Case(
        "MgO Gamma-point calculation",
        mat="MgO_gamma",
    ),
    Case(
        "ZnO",
        mat="ZnO",
    ),
)
def test_init(mat: str) -> None:
    """Tests that pwscf.xml initializes without raising an exception."""
    _ = PWxml(f"tests/data/{mat}/pwscf.xml")


@parametrize_cases(
    Case(
        "MgO Gamma-point calculation",
        mat="MgO_gamma",
        final_energy_eV=-8158.01764759 * Ry_to_eV,
        efermi_eV=5.3759,
    )
)
def test_parsing(mat: str, final_energy_eV: float, efermi_eV: float) -> None:
    """Tests that a pwscf.xml is parsed correctly."""

    pwxml = PWxml(f"tests/data/{mat}/pwscf.xml")
    assert np.allclose(pwxml.final_energy, final_energy_eV)
    assert np.allclose(pwxml.ionic_steps[-1]["total_energy"]["etot"], final_energy_eV)
    assert np.allclose(pwxml.efermi, efermi_eV)
    assert np.allclose(pwxml.vbm, efermi_eV)
