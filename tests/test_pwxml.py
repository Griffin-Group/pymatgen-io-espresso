from pytest_parametrize_cases import Case, parametrize_cases

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
def test_init(mat: str):
    """
    Tests the equality dunder method of Projwfc
    (implicitly also tests the __eq__ method of AtomicState)
    """
    _ = PWxml(f"tests/data/{mat}/pwscf.xml")
