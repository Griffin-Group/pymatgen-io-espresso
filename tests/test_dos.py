from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.espresso.outputs import EspressoDos


@parametrize_cases(
    Case(
        "Si_colinear_nonspinpol",  # Noncolinear, no SOC
        mat="Si",
        lspinorb=False,
        lsda=False,
        noncolinear=False,
        energies_49=-5.367,  # Energy on line 51
        state_3={"l": 1, "m": 3},  # 4th state in projwfc.out
        n_energies=2213,
        pdos_3_1024_up=0.145,
        #  PDOS of up channel at energy 1024 of state 3 (line 1026 of correct pdos file)
        pdos_3_1024_down=None,
    ),
    Case(  # TODO: bad test
        "Ni_colinear_spinpol",  # Noncolinear, no SOC
        mat="Ni",
        lspinorb=False,
        lsda=True,
        noncolinear=False,
        energies_49=-85.818,  # Energy on line 51
        state_3={"l": 1, "m": 3},  # 4th state in projwfc.out
        n_energies=11245,
        pdos_3_1024_up=0.0,
        #  PDOS of up channel at energy 1024 of state 3 (line 1026 of correct pdos file)
        pdos_3_1024_down=0.0,
        #  PDOS of up channel at energy 1024 of state 3 (line 1026 of correct pdos file)
    ),
    Case(
        "Bi2Te3_ncl_nsoc",  # Noncolinear, no SOC
        mat="Bi2Te3",
        lspinorb=False,
        lsda=False,
        noncolinear=True,
        energies_49=-16.045,  # Energy on line 51
        n_energies=2290,
        state_3={"l": 2, "m": 4, "s_z": 0.5},  # 4th state in projwfc.out
        pdos_3_1024_up=0.108e-4,
        # PDOS of up channel at energy 1024 of state 3 (line 1026 of correct pdos file)
        pdos_3_1024_down=None,
    ),
    Case(
        "Sr3PbO_ncl_soc",  # Noncolinear, SOC
        mat="Sr3PbO",
        lspinorb=True,
        lsda=False,
        noncolinear=True,
        energies_49=7.703,  # Energy on line 51
        n_energies=2012,
        state_3={"l": 0, "j": 0.5, "mj": 0.5},  # 4th state in projwfc.out
        pdos_3_1024_up=0.547e-1,
        # PDOS of up channel at energy 1024 of state 3 (line 1026 of correct pdos file)
        pdos_3_1024_down=None,
    ),
)
def test_pdos(
    mat,
    lspinorb,
    lsda,
    noncolinear,
    energies_49,
    n_energies,
    state_3,
    pdos_3_1024_up,
    pdos_3_1024_down,
):
    """
    Tests the equality dunder method of Projwfc
    (implicitly also tests the __eq__ method of AtomicState)
    """
    pdos = EspressoDos.from_filpdos(f"data/{mat}/dos/{mat}")
    assert pdos.lsda == lsda
    assert pdos.lspinorb == lspinorb
    assert pdos.noncolinear == noncolinear
    assert pdos.efermi is None  # Can't read efermi from pdos
    assert pdos.energies[49] == energies_49
    assert len(pdos.energies) == n_energies

    # Test sorting of states
    pdos.atomic_states[7].l == state_3["l"]
    # sourcery skip: no-conditionals-in-tests
    if "s_z" in state_3.keys():
        assert pdos.atomic_states[3].s_z == state_3["s_z"]
    if "m" in state_3.keys():
        assert pdos.atomic_states[3].m == state_3["m"]
    if "j" in state_3.keys():
        assert pdos.atomic_states[3].j == state_3["j"]
    if "m_j" in state_3.keys():
        assert pdos.atomic_states[3].m_j == state_3["m_j"]

    assert (
        pdos.atomic_states[3].pdos[Spin.up][1024] == pdos_3_1024_up
    )  # pdos.keys() == {Spin.up, Spin.down}
    if pdos.lsda:
        assert pdos.atomic_states[3].pdos[Spin.down][1024] == pdos_3_1024_down
