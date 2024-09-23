import numpy as np
import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from pymatgen.core.structure import Structure
from pymatgen.core.units import Ry_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.espresso.outputs import Projwfc


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
        f"tests/data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    projwfc_filproj_dn = Projwfc.from_filproj(
        f"tests/data/{mat}/filproj.projwfc_down", parse_projections=True
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
        f"tests/data/{mat}/projwfc.out", parse_projections=True
    )
    projwfc_filproj_up = Projwfc.from_filproj(
        f"tests/data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    projwfc_filproj_dn = Projwfc.from_filproj(
        f"tests/data/{mat}/filproj.projwfc_down", parse_projections=True
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
    Tests that projwfc.out is parsed correctly
    """
    projwfc_out = Projwfc.from_projwfcout(
        f"tests/data/{mat}/projwfc.out", parse_projections=True
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
        lsda=False,  # Can't detect lsda from filproj up
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
    Tests that filproj is parsed correctly
    """
    projwfc_filproj = Projwfc.from_filproj(
        f"tests/data/{mat}/filproj.projwfc_up", parse_projections=True
    )
    assert projwfc_filproj.nk == nk
    assert projwfc_filproj.nbands == nbands
    assert projwfc_filproj.nstates == nstates
    assert projwfc_filproj.lsda == lsda
    assert projwfc_filproj.lspinorb == lspinorb
    assert projwfc_filproj.noncolin == noncolin

    structure = Structure.from_file(f"tests/data/{mat}/POSCAR")
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
    Tests that filproj is parsed correctly for the spin down channel
    """
    projwfc_filproj = Projwfc.from_filproj(
        f"tests/data/{mat}/filproj.projwfc_down", parse_projections=True
    )
    assert projwfc_filproj.nk == nk
    assert projwfc_filproj.nbands == nbands
    assert projwfc_filproj.nstates == nstates
    assert projwfc_filproj.lsda == lsda
    assert projwfc_filproj.lspinorb == lspinorb
    assert projwfc_filproj.noncolin == noncolin


@parametrize_cases(
    Case(
        "Si",
        mat="Si",
        lsda=False,
        nk=177,
        nbands=8,
        nstates=8,
        kpt1=[-6.25e-02, 0, 0],  # 2nd k-point in the file
        kweight1=1.129943502825000e-002,  # Weight of the 2nd k-point in the file
        eig_kpt1_band4=6.353168731119671e-01
        * Ry_to_eV,  # Energy of band 5 at k-point 2
        efermi=0.453319023952191 * Ry_to_eV,  # Fermi energy
    ),
    Case(
        "Ni_spinpol",
        mat="Ni",
        lsda=True,  # Can't detect lsda from filproj up
        nk=71,
        nbands=10,
        nstates=13,
        kpt1=[-0.025, 0.025, 0.025],  # 2nd k-point in the file
        kweight1=1.408450704225000e-002,  # Weight of the 2nd k-point in the file
        eig_kpt1_band4=6.440717620817992e-01
        * Ry_to_eV,  # Energy of band 5 at k-point 2
        efermi=1.31546784942628 * Ry_to_eV,  # Fermi energy
    ),
    Case(
        "Bi2Te3_nsoc",
        mat="Bi2Te3",
        lsda=False,
        nk=14,
        nbands=144,
        nstates=180,
        kpt1=[0, 0, -6.848264700438211e-02],  # 2nd k-point in the file
        kweight1=1.388888888889000e-002,  # Weight of the 2nd k-point in the file
        eig_kpt1_band4=-1.212380202597821e00
        * Ry_to_eV,  # Energy of band 5 at k-point 2
        efermi=0.464815725414019 * Ry_to_eV,  # Fermi energy
    ),
)
@pytest.mark.filterwarnings("ignore::pymatgen.io.espresso.utils.IbravUntestedWarning")
def test_xml(mat, lsda, nk, nbands, nstates, kpt1, kweight1, eig_kpt1_band4, efermi):
    """
    Tests that filproj is parsed correctly
    """
    projwfc_xml = Projwfc.from_xml(
        f"tests/data/{mat}/atomic_proj.xml", parse_projections=True
    )
    assert projwfc_xml.nk == nk
    assert projwfc_xml.nbands == nbands
    assert projwfc_xml.nstates == nstates
    assert projwfc_xml.lsda == lsda
    # Can't detect lspinorb/noncolin from atomic_proj.xml
    assert projwfc_xml.lspinorb is None
    assert projwfc_xml.noncolin is None

    assert np.all(projwfc_xml.k[1] == kpt1)
    assert projwfc_xml.k_weights[1] == kweight1
    assert projwfc_xml.eigenvals[Spin.up][1][4] == eig_kpt1_band4
    assert projwfc_xml.parameters["efermi"] == efermi
