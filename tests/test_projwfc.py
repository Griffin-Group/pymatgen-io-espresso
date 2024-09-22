from pymatgen.io.espresso.outputs import Projwfc
from pymatgen.electronic_structure.core import Spin
import numpy as np
import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.core.structure import Structure


@parametrize_cases(
    Case(
        "Ni_spinpol",
        mat="Ni",
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_eq(mat):
    """
    Tests the equality dunder method of Projwfc
    (implicitly also tests the __eq__ method of AtomicState)
    """
    projwfc_filproj_up = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    projwfc_filproj_dn = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_down", parse_projections=True
    )
    assert projwfc_filproj_up == projwfc_filproj_dn


@parametrize_cases(
    Case(
        "Ni_spinpol",
        mat="Ni",
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_add(mat):
    """
    Tests the addition dunder method of Projwfc, and also compares
    Projwfc.from_filproj with Projwfc.from_projwfcout
    """
    projwfc_out = Projwfc.from_projwfcout(
        f"data/{mat}/projwfc.out", parse_projections=True
    )
    projwfc_filproj_up = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    projwfc_filproj_dn = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_down", parse_projections=True
    )
    projwfc_filproj = projwfc_filproj_up + projwfc_filproj_dn
    # sourcery skip: no-loop-in-tests
    for state_idx in range(projwfc_out.nstates):
        for spin in [Spin.up, Spin.down]:
            p1 = projwfc_out.atomic_states[state_idx].projections[spin]
            # projwfc.out only has three decimal places
            p2 = np.around(
                projwfc_filproj.atomic_states[state_idx].projections[spin],
                decimals=3,
            )

            assert np.allclose(p1, p2, atol=1e-3, rtol=0)


@parametrize_cases(
    Case(
        "Si",
        mat="Si",
        lsda=False,
        lspinorb=False,
        noncolin=False,
        nk=177,
        nbands=8,
        nstates=8,
    ),
    Case(
        "Ni_spinpol",
        mat="Ni",
        lsda=True,
        lspinorb=False,
        noncolin=False,
        nk=71,
        nbands=10,
        nstates=13,
    ),
    Case(
        "Bi2Te3_nsoc",
        mat="Bi2Te3",
        lsda=False,
        lspinorb=False,
        noncolin=True,
        nk=14,
        nbands=144,
        nstates=180,
    ),
    Case(
        "Sr3PbO_soc",
        mat="Sr3PbO",
        lsda=False,
        lspinorb=True,
        noncolin=True,
        nk=234,
        nbands=96,
        nstates=104,
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_projwfcout(mat, lsda, lspinorb, noncolin, nk, nbands, nstates):
    """
    Tests the addition dunder method of Projwfc, and also compares
    Projwfc.from_filproj with Projwfc.from_projwfcout
    """
    projwfc_out = Projwfc.from_projwfcout(
        f"data/{mat}/projwfc.out", parse_projections=True
    )
    assert projwfc_out.nk == nk
    assert projwfc_out.nbands == nbands
    assert projwfc_out.nstates == nstates
    assert projwfc_out.lsda == lsda
    assert projwfc_out.lspinorb == lspinorb
    assert projwfc_out.noncolin == noncolin

@parametrize_cases(
    Case(
        "Si",
        mat="Si",
        lsda=False,
        lspinorb=False,
        noncolin=False,
        nk=177,
        nbands=8,
        nstates=8,
    ),
    Case(
        "Ni_spinpol",
        mat="Ni",
        lsda=False,
        lspinorb=False,
        noncolin=False,
        nk=71,
        nbands=10,
        nstates=13,
    ),
    Case(
        "Bi2Te3_nsoc",
        mat="Bi2Te3",
        lsda=False,
        lspinorb=False,
        noncolin=True,
        nk=14,
        nbands=144,
        nstates=180,
    ),
    Case(
        "Sr3PbO_soc",
        mat="Sr3PbO",
        lsda=False,
        lspinorb=True,
        noncolin=True,
        nk=234,
        nbands=96,
        nstates=104,
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_filproj(mat, lsda, lspinorb, noncolin, nk, nbands, nstates):
    """
    Tests the addition dunder method of Projwfc, and also compares
    Projwfc.from_filproj with Projwfc.from_projwfcout
    """
    projwfc_filproj = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    assert projwfc_filproj.nk == nk
    assert projwfc_filproj.nbands == nbands
    assert projwfc_filproj.nstates == nstates
    assert projwfc_filproj.lsda == lsda
    assert projwfc_filproj.lspinorb == lspinorb
    assert projwfc_filproj.noncolin == noncolin

    structure = Structure.from_file(f"data/{mat}/POSCAR")
    for site in projwfc_filproj.structure:
        site.properties = {}
    assert projwfc_filproj.structure == structure


@parametrize_cases(
    Case(
        "Ni_spinpol",
        mat="Ni",
        lsda=True,
        lspinorb=False,
        noncolin=False,
        nk=71,
        nbands=10,
        nstates=13,
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_filproj_down(mat, lsda, lspinorb, noncolin, nk, nbands, nstates):
    """
    Tests the addition dunder method of Projwfc, and also compares
    Projwfc.from_filproj with Projwfc.from_projwfcout
    """
    projwfc_filproj = Projwfc.from_filproj(
        f"data/{mat}/filproj.projwfc_down", parse_projections=True
    )
    assert projwfc_filproj.nk == nk
    assert projwfc_filproj.nbands == nbands
    assert projwfc_filproj.nstates == nstates
    assert projwfc_filproj.lsda == lsda
    assert projwfc_filproj.lspinorb == lspinorb
    assert projwfc_filproj.noncolin == noncolin
