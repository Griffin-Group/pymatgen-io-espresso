"""
Classes for reading/manipulating PWscf xml files.
"""

import os
import re
import warnings
from collections import defaultdict
from typing import Literal, TextIO

import numpy as np
import xmltodict
from monty.io import zopen

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.units import (
    Ha_to_eV,
    bohr_to_ang,
    unitized,
)
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.espresso.inputs.pwin import KPointsCard, PWin
from pymatgen.io.espresso.outputs.dos import EspressoDos
from pymatgen.io.espresso.outputs.projwfc import (
    AtomicState,
    InconsistentProjwfcDataError,
    Projwfc,
)
from pymatgen.io.espresso.utils import (
    FileGuesser,
    parse_pwvals,
    projwfc_orbital_to_vasp,
)
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun


# TODO: write docstring
class PWxml(Vasprun):
    r"""
    Parser for PWscf xml files. Almost all Vasprun features are implemented. This class
    exposes a public API that is practically identical to the Vasprun class, with values
    always returned in the same coordinate and unit systems used by VASP. Can be used
    interchangeably with Vasprun for most purposes with no changes to the calling code. This class will pull data from the output of `projwfc.x` and `dos.x` when necessary, attempting to guess the filenames if they are not provided.

    Some attributes are not particularly meaningful for QE, and a few are not implemented.

    Missing Vasprun features:

    * Projected magnetization. Not sure we can get this from QE (maybe `pp.x`?).
        `bands.x` supports computing $\langle \sigma_\alpha \rangle$, but not
        projected.
    * Phononic properties: Need to write parser for `ph.x` output for dynamical matrix,
        interatomic force constants, phonon frequencies, and eigenvectors.
    * Dielectric properties: None of these are implemented. Some can be obtained from
      `ph.x` or QE's old `epsilon.x`. Some are RPA and thus not available in QE.

    Not all attributes are listed below since many are inherited directly from Vasprun.

    Attributes:
        kpoints_frac (np.ndarray): kpoints in fractional coordinates, (nk, 3) array
        kpoints_cart (np.ndarray): kpoints in cartesian coords (1/angst), (nk, 3) array
        actual_kpoints (np.ndarray): Same as kpoints_frac, maintained for
            compatibility with Vasprun.
        actual_kpoints_weights (np.ndarray): List of kpoint weights, normalized.
        atomic_symbols (list[str]): List of atomic symbols, e.g., ["Li", "Fe", "Fe", "P", "P", "P"].
        efermi (float): Fermi energy, eV.
        vbm (float): Valence band maximum, as computed by QE, in eV.
        cbm (float): Conduction band minimum, as computed by QE, in eV.
        eigenvalues (dict[Spin, np.ndarray]): Final eigenvalues (in eV) as
            a dict with keys Spin.up/Spin.down, and the values are numpy arrays
            with dimensions (nk, nb, 2), where nk is the number of kpoints, nb is
            the number of bands, and the last axis is the eigenvalue and occupation.
            Identical representation to Vasprun.
        projected_eigenvalues (dict[Spin, np.ndarray]): Final projected
            eigenvalues as a dict of {spin: nd-array}. To access a particular value, you need to index as `[spin][kpoint index][band index][atom index][orbital_index]`. This representation is identical to Vasprun.xml.
        tdos (Dos): Total dos from fildos or filpdos. filpdos overrides fildos.
        idos (Dos): Integrated dos from fildos.
        pdos ([List[dict[Orbital, Dict[spin, np.ndarray]]]]):
            List of Dos objects, indexed as `[atom][orbital][spin]`.
        ionic_steps (list[dict]): All ionic steps, list of

            {
            "structure": structure at end of run,
            "total_energy": {"ehart", "etxc", ...}
            "stress": pressure in eV/A3,
            "forces": forces in eV/A
            }.

        nionic_steps (int): The total number of ionic steps. This number is always
            equal to the total number of steps in the actual run even if
            ionic_step_skip is used. nionic_steps here has a slightly different meaning
            from the Vasprun class. VASP will first do an SCF calculation with the
            input structure, then perform geometry optimization until you hit EDIFFG
            or NSW, then it's done. QE does the same thing for relax, but it will also
            do a final SCF calculation with the optimized structure and a new basis set
            for vc-relax. In reality, converged QE vc-relax calculations take
            nionic_steps-1 to converge.
        projected_magnetisation (dict[Spin, np.ndarray]): Not implemented
        other_dielectric (dict): Not implemented
        force_constants (np.ndarray): Not implemented.
        normalmode_eigenvals (np.ndarray): Not implemented.
        normalmode_eigenvecs (np.ndarray): Not implemented.
        md_data (dict): Molecular dynamics data, not implemented for QE.
        parameters (dict[str, Any]): Parameters of the PWscf run from the XML.
             Naturally very different from Vasprun.
        incar (Incar): Empty Incar object, for compatibility.
        kpoints (Kpoints): Empty Kpoints object, for compatibility.
        pseudo_filenames (list[str]): List of pseudopotential filenames,
            e.g., ["Li.pbe-spn-kjpaw_psl.0.1.UPF", "Fe.pbe-n-kjpaw_psl.0.2.1.UPF", ...].
        potcar_symbols: Maintained for compatibility with Vasprun, has Psuedo
            + element names, e.g., ['Si.pbesol-n-rrkjus_psl.1.0.0.UPF Si'].

    Author: Omar A. Ashour


    """

    def __init__(
        self,
        filename: str | os.PathLike,
        ionic_step_skip: int = 1,
        ionic_step_offset: int = 0,
        parse_dos: bool = False,
        fildos: str | os.PathLike | None = None,
        parse_pdos: bool = False,
        filpdos: str | os.PathLike | None = None,
        parse_projected_eigen: bool = False,
        filproj: str | os.PathLike | None = None,
        occu_tol: float = 1e-8,
        separate_spins: bool = False,
        **_,  # Ignored arguments for compatibility with Vasprun
    ):
        """
        Constructor for the PWxml class.

        Args:
            filename (str | os.PathLike): Path to XML file to parse
            ionic_step_skip (int): If ionic_step_skip is a number > 1,
                only every ionic_step_skip ionic steps will be read for
                structure and energies. Unlike Vasprun, the final energy
                will always be the total energy of the scf calculation
                performed after ionic convergence. This is not very useful
                since PWscf xml files aren't as huge as Vasprun files.
                Mainly kept for consistency with the Vasprun class.
            ionic_step_offset (int): Used together with ionic_step_skip. If set,
                the first ionic step read will be offset by the amount of
                ionic_step_offset. For example, if you want to start reading
                every 10th structure but only from the 3rd structure onwards,
                set ionic_step_skip to 10 and ionic_step_offset to 3. Main use
                case is when doing statistical structure analysis with
                extremely long time scale PWscf calculations of
                varying numbers of steps, and kept for consistency with the
                Vasprun class.
            parse_dos (bool): Whether to parse the dos. If True, PWxml will use
                fildos if provided, or attempt to guess the filename.
            fildos (str): If provided, forces parse_dos to be True and uses the
                provided string as the path to the dos file. This is
                the same as in dos.x input, and shouldn't be a full path to a file.
                For example, filpdos="path/to/filpdos" will look for
                "path/to/filpdos.pdos_*"
            filpdos (str): If provided, forces parse_dos to be True and uses
                 the provided string as 'filproj', same as in projwfc.x input.
                 It shouldn't include the rest of the filename. For example,
                 filpdos="path/to/filpdos" will look for "path/to/filpdos.pdos_*"
            parse_projected_eigen (bool): Whether to parse the projected
                eigenvalues and (magnetisation, not implemented). Defaults to False.
                If True, PWxml will look for a "filproj" from projwfc.x and parse it.
                It will look for files with the same name as the XML (or same QE prefix)
                but with a .projwfc_up extension, or will use the filproj argument
            filproj (str): If provided, forces parse_projected_eigen to be True and
                uses the provided string as the filepath to the .projwfc_up file.
                Note that this is the same as in projwfc.x input, so it shouldn't
                include the .projwfc_up/down extension. It can also include
                a directory, e.g., "path/to/filproj" will look for
                "path/to/filproj.projwfc_up"
            occu_tol (float): Sets the minimum tol for the determination of the
                vbm and cbm. Usually the default of 1e-8 works well enough,
                but there may be pathological cases. Note that, unlike VASP, QE
                actually reports the VBM and CBM (accessible via the vbm and cbm
                attributes), so this is only used to recompute them
                and check against the reported values.
            separate_spins (bool): Whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
        """
        self._filename = filename
        self.ionic_step_skip = ionic_step_skip
        self.ionic_step_offset = ionic_step_offset
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins

        if filproj:
            parse_projected_eigen = True
        if fildos:
            parse_dos = True
        if filpdos:
            parse_pdos = True

        with zopen(filename, "rt") as f:
            self._parse(
                f,
                parse_dos=parse_dos,
                fildos=fildos,
                parse_pdos=parse_pdos,
                filpdos=filpdos,
                parse_projected_eigen=parse_projected_eigen,
                filproj=filproj,
                ionic_step_skip=ionic_step_skip,
                ionic_step_offset=ionic_step_offset,
            )

        if not self.converged:
            warnings.warn(
                (
                    f"{filename} is an unconverged PWscf run.\n"
                    f"Electronic convergence reached: {self.converged_electronic}.\n"
                    f"Ionic convergence reached: {self.converged_ionic}."
                ),
                UnconvergedPWxmlWarning,
            )

    def _parse(
        self,
        stream: TextIO,
        parse_dos: bool,
        fildos: str | os.PathLike | None,
        parse_pdos: bool,
        filpdos: str | os.PathLike | None,
        parse_projected_eigen: bool,
        filproj: str | os.PathLike | None,
        ionic_step_skip: int,
        ionic_step_offset: int,
    ):
        """
        Parses the xml file. See __init__ docstring for documentation of the arguments.
        """
        # Very useful for debugging and accessing unparsed data
        self._raw_dict = xmltodict.parse(stream.read())["qes:espresso"]

        input_section = self._raw_dict["input"]
        output_section = self._raw_dict["output"]
        b_struct = output_section["band_structure"]

        # Sets some useful attributes
        self._set_pw_calc_params(self._raw_dict, input_section, b_struct)

        self.initial_structure = self._parse_structure(
            input_section["atomic_structure"]
        )
        self.atomic_symbols = [s.species_string for s in self.initial_structure]
        self.pseudo_elements, self.pseudo_filenames = self._parse_atominfo(
            input_section["atomic_species"]
        )

        # Deal with the ionic steps and relaxation trajectory
        self.final_structure, self.ionic_steps = self._parse_relaxation(
            ionic_step_skip, ionic_step_offset, self._raw_dict, output_section
        )
        # TODO: check this works correctly, I think in Vasprun it
        # returns the full number of steps regardless of skip/offset.
        # My implementation used to work that way but I changed it during a refactor
        self.nionic_steps = len(self.ionic_steps)

        # Band structure and some properties
        self.eigenvalues = self._parse_eigen(b_struct["ks_energies"], self.lsda)
        self.efermi, self.vbm, self.cbm = self._parse_vbm_cbm_efermi(b_struct)

        # k-points
        self.alat = parse_pwvals(output_section["atomic_structure"]["@alat"])
        # Transformation matrix from cart. to frac. coords. in k-space
        T = self.final_structure.lattice.reciprocal_lattice.matrix
        T = np.linalg.inv(T).T
        self.kpoints_frac, self.kpoints_cart, self.actual_kpoints_weights = (
            self._parse_kpoints(output_section, T, self.alat)
        )
        self.actual_kpoints = self.kpoints_frac

        # Projected eigenvalues, dos, pdos, etc.
        if parse_projected_eigen:
            if filproj is None:
                filproj = FileGuesser("filproj", self._filename, self.prefix).guess()
            self.atomic_states = (
                self._parse_projected_eigen(filproj) if parse_projected_eigen else None
            )

        if parse_dos:
            if fildos is None:
                fildos = FileGuesser("fildos", self._filename, self.prefix).guess()
            self.tdos, self.idos = self._parse_dos(fildos)
        if parse_pdos:
            if filpdos is None:
                filpdos = FileGuesser("filpdos", self._filename, self.prefix).guess()
            self.tdos, self.pdos = self._parse_pdos(filpdos)

        self._fudge_vasp_params()

    @property
    def is_spin(self) -> bool:
        """
        Returns:
            True if the calculation is spin-polarized.
        """
        return self.lsda

    @property
    def converged_electronic(self) -> bool:
        """
        Returns true if electronic convergence was reached in the last ionic step. Note that, even though QE internally considers NSCF calculations unconverged, this property is set to True for NSCF (and bands) calcs to maintain consistency with the Vasprun class.

        Returns:
            True if electronic step convergence has been reached in the final
            ionic step
        """
        if self.parameters["control_variables"]["calculation"] in ("nscf", "bands"):
            return True
        return self.ionic_steps[-1]["scf_conv"]["convergence_achieved"]

    @property
    def converged_ionic(self) -> bool:
        """
        Returns:
            True if ionic step convergence has been reached

        To maintain consistency with the Vasprun class, we return True if the calculation didn't involve geometric optimization (scf, nscf, ...)
        """
        # Check if dict has 'ionic_conv' key
        if "ionic_conv" in self.ionic_steps[-1]:
            return self.ionic_steps[-1]["ionic_conv"]["convergence_achieved"]
        return True

    @property
    @unitized("eV")
    def final_energy(self) -> float:
        """
        Final energy from the PWscf run, in eV.
        """
        final_istep = self.ionic_steps[-1]
        total_energy = final_istep["total_energy"]["etot"]
        if total_energy == 0:
            warnings.warn(
                "Calculation has zero total energy. Possibly an NSCF or bands run.",
                ZeroTotalEnergyWarning,
            )
        return total_energy * Ha_to_eV

    @property
    def hubbards(self) -> dict[str, float]:
        """
        Hubbard U values used if a vasprun is a GGA+U run. {} otherwise.

        Returns:
            dict: {element: U_value}
        """
        # TODO: check if U stuff changed in QE v7.2
        if self.parameters["dft"].get("dftU", False):
            U_list = self.parameters["dft"]["dftU"]["Hubbard_U"]  # type: ignore
            return {U["@specie"] + ":" + U["@label"]: U["#text"] for U in U_list}  # type: ignore
        return {}

    @property
    def run_type(self):
        """
        Returns the run type.

        Should be able to detect functional, Hubbard U terms and vdW corrections.
        """
        rt = self.parameters["dft"]["functional"]
        # TODO: check if U stuff changed in QE v7.2
        if self.parameters["dft"].get("dftU", False):
            if self.parameters["dft"]["dftU"]["lda_plus_u_kind"] == 0:
                rt += "+U"
            elif self.parameters["dft"]["dftU"]["lda_plus_u_kind"] == 1:
                rt += "+U+J"
            else:
                rt += "+U+?"
        if self.parameters["dft"].get("vdW", False):
            rt += "+" + self.parameters["dft"]["vdW"]["vdw_corr"]
        return rt

    def get_band_structure(
        self,
        kpoints_filename: str | None = None,
        efermi: float | Literal["smart"] | None = "smart",
        line_mode: bool = False,
        force_hybrid_mode: bool = False,
    ) -> BandStructureSymmLine | BandStructure:
        """Get the band structure as a BandStructure object.

        Args:
            kpoints_filename: Path of the PWscf input file from which the band
                structure is generated. If none is provided, the code will try to
                guess the appropriate file, see the docstring of `FileGuesser`.
            efermi: The Fermi energy associated with the bandstructure, in eV. By
                default (None), uses the value reported by PWscf in the xml. To
                manually set the Fermi energy, pass a float. Pass 'smart' to use the
                `calculate_efermi()` method, which is identical for metals but more
                accurate for insulators (mid-gap).
            line_mode: Force the band structure to be considered as
                a run along symmetry lines. (Default: False)
            force_hybrid_mode: Not Yet Implemented (Default: False)

        Returns:
            a BandStructure object (or more specifically a
                BandStructureSymmLine object if the run is detected to be a run
                along symmetry lines) NSCF (calc='nscf' or 'bands') calculations are accepted for Line-Mode with explicit PWscf input file, and 'crystal', 'crystal_b', 'tpiba' or 'tpiba_b' K_POINTS card. The k-points needs to have data on the kpoint label as a comment.
        """
        factor = 1 if self.noncolin else 2
        if self.nbands <= self.nelec / factor:
            warnings.warn(
                f"Number of bands ({self.nbands}) <= number of electrons/{factor} "
                f"({self.nelec / factor:.4f}). BSPlotter may not work properly.",
                DifferentFromVASPWarning,
            )

        if not kpoints_filename:
            kpoints_filename = FileGuesser("pwin", self._filename, self.prefix).guess()

        if kpoints_filename and not os.path.exists(kpoints_filename) and line_mode:
            raise PWxmlParserError(
                "PW input file needed to obtain band structure along symmetry lines."
            )

        if efermi == "smart":
            e_fermi = self.calculate_efermi()
        elif efermi is None:
            e_fermi = self.efermi
        else:
            e_fermi = efermi

        lattice_new = Lattice(self.final_structure.lattice.reciprocal_lattice.matrix)

        p_eigenvals: defaultdict[Spin, list] = defaultdict(list)
        eigenvals: defaultdict[Spin, list] = defaultdict(list)

        for spin, v in self.eigenvalues.items():
            v = np.swapaxes(v, 0, 1)
            eigenvals[spin] = v[:, :, 0]

            if self.projected_eigenvalues:
                peigen = self.projected_eigenvalues[spin]
                # Original axes for self.projected_eigenvalues are kpoints,
                # band, ion, orb.
                # For BS input, we need band, kpoints, orb, ion.
                peigen = np.swapaxes(peigen, 0, 1)  # Swap kpoint and band axes
                peigen = np.swapaxes(peigen, 2, 3)  # Swap ion and orb axes

                p_eigenvals[spin] = peigen

        k_card = None
        if kpoints_filename and os.path.exists(kpoints_filename):
            k_card = PWin.from_file(kpoints_filename).k_points
        coords_are_cartesian = False if k_card is None else k_card.coords_are_cartesian
        if coords_are_cartesian:
            kpoints = [np.array(kpt) for kpt in self.kpoints_cart]
        else:
            kpoints = [np.array(kpt) for kpt in self.kpoints_frac]

        if k_card.line_mode or line_mode:
            labels_dict = {}
            # TODO: check how hybrid band structs work in QE
            hybrid_band = False
            if hybrid_band or force_hybrid_mode:
                raise NotImplementedError(
                    "Hybrid band structures not yet supported in line mode."
                )
            kpoints, eigenvals, p_eigenvals, labels_dict = self._vaspify_kpts_bands(
                kpoints, eigenvals, p_eigenvals, k_card, self.alat
            )
            return BandStructureSymmLine(
                kpoints,
                eigenvals,
                lattice_new,
                e_fermi,
                labels_dict,
                structure=self.final_structure,
                projections=p_eigenvals,
                coords_are_cartesian=coords_are_cartesian,
            )
        return BandStructure(
            kpoints,
            eigenvals,
            lattice_new,
            e_fermi,
            structure=self.final_structure,
            projections=p_eigenvals,
            coords_are_cartesian=coords_are_cartesian,
        )

    @property
    def eigenvalue_band_properties(
        self,
    ) -> (
        tuple[float, float, float, bool]
        | tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[bool, bool],
        ]
    ):
        """
        Returns the band gap, CBM, VBM, and whether the gap is direct. Directly
        uses the Vasprun implementation, with some extra check against the
        CBM and VBM values reported by QE.

        Returns:
            If separate_spins is False, returns a tuple of the
                band gap, CBM, VBM, and whether the gap is direct.
                If separate_spins is True, returns a tuple of tuples of the
                band gap, CBM, VBM, and whether the gap is direct for each spin channel.
        """
        gap, cbm_pmg, vbm_pmg, direct = super().eigenvalue_band_properties

        cbm = np.amin(cbm_pmg) if self.separate_spins else cbm_pmg
        vbm = np.amax(vbm_pmg) if self.separate_spins else vbm_pmg

        if self.vbm and not np.isclose(vbm, self.vbm, atol=1e-3):
            warnings.warn(
                f"VBM computed by pw.x is different from the one computed by pymatgen. "
                f"(difference = {np.abs(vbm - self.vbm) * 1000:.2f} meV)."
            )
        if self.cbm and not np.isclose(cbm, self.cbm, atol=1e-3):
            warnings.warn(
                f"CBM computed by pw.x is different from the one computed by pymatgen. "
                f"(difference = {np.abs(cbm - self.cbm) * 1000:.2f} meV)."
            )
        return gap, cbm_pmg, vbm_pmg, direct

    def calculate_efermi(self, **_) -> float:
        """
        Calculate the Fermi level

        PWscf returns the Fermi level for all calculations and the cbm and vbm for
        all insulators. These are stored in PWxml.efermi, PWxml.cbm, and PWxml.vbm.
        However, for insulators, the Fermi level is often slightly off from the
        exact mid-gap value.

        If vbm and cbm are both undefined (metallic system), return the Fermi level
        if vbm is defined and cbm isn't, it's usually a sign of an insulator
        with as many bands as electrons (often nbnd isn't set in input)
        Such calculations don't work with BSPlotter()

        Returns:
            The Fermi level, in eV
        """
        if self.vbm is None or self.cbm is None:
            return self.efermi
        return (self.vbm + self.cbm) / 2

    @property
    def projected_eigenvalues(self) -> dict[Spin, np.ndarray] | None:
        """
        Returns the projected eigenvalues in the same format Vasprun uses
        (i.e., the VASP convention)

        In the case of SOC, QE uses the |LJJz> basis while VASP uses the |LLz> basis.
        This function will sum all p states into where py should be (index 1) and all d
        states into where dxy should be (index 4). The rest will be 0.

        # TODO: cleanup and rewrite
        """
        if self.atomic_states is None:
            return None

        if self.lspinorb:
            # NOTE: probably not worth implementing this with Clebsch-Gordan.
            # Probably better to write a parser for the W90 A_mn output...
            warnings.warn(
                "Quantum espresso works in the |LJJz> basis when SOC is enabled while "
                "VASP uses the |LLz> basis. Converting between the two is not "
                "currently implemented. projected_eigenvalues will have all "
                "p states summed into where py should be (index 1) and all d states "
                "summed into where dxy should be (index 4). The rest will be 0. ",
                DifferentFromVASPWarning,
            )
        elif self.noncolin:
            # TODO: implement this
            warnings.warn(
                "Lz resolution for noncolinear calculations is not currently "
                "implemented. projected_eigenvalues will have all p states summed "
                "into where py should be (index 1) and all d states summed into where "
                "dxy should be (index 4). The rest will be 0. ",
                DifferentFromVASPWarning,
            )

        projected_eigenvalues = {
            Spin.up: np.zeros(
                (self.nk, self.nbands, self.initial_structure.num_sites, 9)
            )
        }
        if self.lsda:
            projected_eigenvalues[Spin.down] = np.zeros(
                (self.nk, self.nbands, self.initial_structure.num_sites, 9)
            )
        for state in self.atomic_states:
            for spin in state.projections.keys():
                if self.lspinorb or self.noncolin:
                    orbital_i = projwfc_orbital_to_vasp(state.l, 2 * state.l + 1)
                    projected_eigenvalues[spin][
                        :, :, state.site.atom_i - 1, orbital_i
                    ] += state.projections[spin]
                else:
                    projected_eigenvalues[spin][
                        :, :, state.site.atom_i - 1, state.orbital.value
                    ] = state.projections[spin]

        return projected_eigenvalues

    @property
    def projected_magnetisation(self):
        """
        Returns the projected magnetisation for each atom in a format compatible with
        the Vasprun class. Not currently implemented for QE.
        """
        # NOTE: QE does not actually provide this information as far as I know.
        warnings.warn(
            "Projected magnetisation not implemented for QE. Returning None.",
            NotImplementedForQE,
        )
        return None

    @property
    def md_data(self):
        """
        Molecular dynamics data. Not currently implemented for QE.
        """
        warnings.warn(
            "MD data not implemented for QE. Returning None.", NotImplementedForQE
        )
        return None

    @property
    def epsilon_static(self) -> list[float]:
        """
        The static part of the dielectric constant. Not implemented for QE.
        """
        warnings.warn(
            "Static dielectric constant not implemented for QE. Returning None.",
            NotImplementedForQE,
        )
        return None

    @property
    def epsilon_static_wolfe(self) -> list[float]:
        """
        The static part of the dielectric constant without any local
        field effects. Not implemented for QE.
        """
        warnings.warn(
            "Static dielectric constant without local field effects "
            "not implemented for QE. Returning None.",
            NotImplementedForQE,
        )
        return None

    @property
    def epsilon_ionic(self) -> list[float]:
        """
        The ionic part of the static dielectric constant. Not implemented for QE.
        """
        warnings.warn(
            "Ionic part of the static dielectric constant not implemented for QE. "
            "Returning None.",
            NotImplementedForQE,
        )
        return None

    @property
    def dielectric(self) -> tuple[list, list, list]:
        """The real and imaginary part of the dielectric constant (e.g.,
        computed by RPA) in function of the energy (frequency).
        Optical properties (e.g. absorption coefficient) can be obtained through this.

        Not implemented for QE.
        """
        warnings.warn(
            "Dielectric constant not implemented for QE. Returning None.",
            NotImplementedForQE,
        )
        return None

    @property
    def optical_absorption_coeff(self) -> list[float] | None:
        """
        The optical absorption coefficient from the dielectric constants.
        Not implemented for QE.
        """
        warnings.warn(
            "Optical absorption coefficient not implemented for QE. Returning None.",
            NotImplementedForQE,
        )
        return None

    def _parse_projected_eigen(self, filproj):
        """
        Parse the projected eigenvalues from a projwfc.x filproj file.

        # TODO: cleanup and rewrite
        """
        projwfc = Projwfc.from_filproj(f"{filproj}.projwfc_up")
        self._validate_filproj(projwfc)
        if self.lsda:
            projwfc_down = Projwfc.from_filproj(f"{filproj}.projwfc_down")
            try:
                projwfc = projwfc + projwfc_down
            except InconsistentProjwfcDataError as e:
                raise InconsistentWithXMLError(
                    f"Error in combining projwfc.x files for spin up/down: {e}"
                ) from e

        return projwfc.atomic_states

    def _validate_filproj(self, p):
        """
        Validates that the Projwfc object is consistent with the PWxml object.

        # TODO: cleanup and rewrite
        """
        if p.nk != self.nk:
            raise InconsistentWithXMLError(
                f"Number of kpoints in {self._filename} ({self.nk}) and "
                "{p._filename} ({p.nk}) do not match."
            )
        if p.nbands != self.nbands:
            raise InconsistentWithXMLError(
                f"Number of bands in {self._filename} ({self.nbands}) and "
                f"{p._filename} ({p.nbands}) do not match."
            )
        if p.lspinorb != self.lspinorb:
            raise InconsistentWithXMLError(
                f"lsorbit in {self._filename} ({self.lspinorb}) and "
                f"{p._filename} ({p.lspinorb}) do not match."
            )
        if p.noncolin != self.noncolin:
            raise InconsistentWithXMLError(
                f"noncolin in {self._filename} ({self.noncolin}) and "
                f"{p._filename} ({p.noncolin}) do not match."
            )
        # TODO: currently not testing whole structure in case of precision
        # issues and to avoid dealing with site properties. Should implement.
        if p.structure.num_sites != self.initial_structure.num_sites:
            raise InconsistentWithXMLError(
                f"Number of atoms in {self._filename} "
                f"({self.initial_structure.num_sites}) and "
                f"{p._filename} ({p.structure.num_sites}) do not match."
            )

    @staticmethod
    def _vaspify_kpts_bands(
        kpoints: np.ndarray,
        eigenvals: dict[Spin, np.ndarray],
        p_eigenvals: dict[Spin, np.ndarray],
        k_card: KPointsCard,
        alat: float,
    ) -> tuple[
        np.ndarray,
        dict[Spin, np.ndarray],
        dict[Spin, np.ndarray],
        dict[np.ndarray, str],
    ]:
        """
        Helper function to convert kpoints and eigenvalues to the format
        expected by the BandStructureSymmLine class.

        VASP duplicates k-points along symmetry lines, while QE does not.
        For example, if you do a BS calculation along the path
        X - G - X, VASP will do X - more kpts - G - G - more kpts - X, while QE will do
        X - more kpts - G - more kpts - X. This function duplicates HSPs so that
        BandStructureSymmLine works properly.

        Args:
            kpoints: kpoints in fractional or cartesian coordinates, (nk, 3) array
            eigenvals: Eigenvalues (in eV) as a dict with keys Spin.up/Spin.down,
                and the values are numpy arrays with dimensions (nk, nb), where nk is
                the number of kpoints, nb is the number of bands.
            p_eigenvals: Projected eigenvalues as a dict of {spin: nd-array}, indexed as
                `[spin][kpoint index][band index][atom index][orbital_index].`
            k_card: KPointsCard object from the PWscf input file
            alat: Lattice constant from the XML, in angstrom

        Returns:
            kpoints: kpoints, with duplicates added
            eigenvals: Eigenvalues, with duplicates added
            p_eigenvals: Projected eigenvalues, with duplicates added
            labels_dict: Dictionary of kpoint labels
        """
        labels = k_card.labels
        factor = (
            (2 * np.pi / alat) * (1 / bohr_to_ang) if k_card.coords_are_cartesian else 1
        )
        input_kpoints = np.array(k_card.k) * factor
        nkpts = k_card.weights
        if k_card.band_mode and "" in labels:
            labels = [label if label != "" else "?" for label in labels]
            warnings.warn(
                "Unlabelled k-point(s) found in input file in band (*_b) mode, "
                "replacing with '?'."
            )
        labels_dict = dict(zip(labels, input_kpoints))
        labels_dict.pop("", None)

        # Figure out the indices of the HSPs that require duplication
        if k_card.band_mode:
            # pw.x doesn't read the weight of the last k-point, it's treated as just 1
            nkpts[-1] = 1
            nkpts.insert(0, 0)
            hsp_idx = np.cumsum(nkpts)
        else:
            hsp_idx = np.where(np.array(labels) != "")[0][:-1]
        # HSPs with consecutive indices occur at discontinuties,
        # they don't need duplication This also takes care of last HSP with *_b options
        discont_idx = np.where(np.diff(hsp_idx) == 1)[0]
        discont_idx = np.concatenate((discont_idx, discont_idx + 1))
        hsp_idx = np.delete(hsp_idx, discont_idx)
        # Start of path doesn't need duplication
        hsp_idx = hsp_idx[1:]

        for i, idx in enumerate(hsp_idx):
            kpoints = np.insert(kpoints, idx + i + 1, kpoints[idx + i], axis=0)
            for spin in eigenvals:
                eigenvals[spin] = np.insert(
                    eigenvals[spin], idx + i + 1, eigenvals[spin][:, idx + i], axis=1
                )
                if p_eigenvals:
                    p_eigenvals[spin] = np.insert(
                        p_eigenvals[spin],
                        idx + i + 1,
                        p_eigenvals[spin][:, idx + i, :, :],
                        axis=1,
                    )

        return kpoints, eigenvals, p_eigenvals, labels_dict

    def _parse_dos(self, fildos):
        """
        Parses the density of states from the output of projwfc.x and/or dos.x

        # TODO: cleanup and rewrite
        """
        dos = EspressoDos.from_fildos(fildos)

        if (self.noncolin and not self.lspinorb) or self.lsda:
            warnings.warn(
                "Detected spin-polarized colinear calculation or noncolinear "
                "calculation without SOC and using fildos. The integrated DOS "
                " will only have one spin channel that includes both spin up "
                "and spin down. This is due to differences in QE and VASP outputs."
            )
        # With SOC VASP includes a spin down channel for TDOS and IDOS
        # For IDOS it's purely zeros, for TDOS it has some small values,
        # not sure what they are, but they don't seem to be used for anything?
        dos.idos = {Spin.up: dos.idos}
        if self.lspinorb:
            dos.tdos[Spin.down] = np.zeros_like(dos.tdos[Spin.up])
            dos.idos[Spin.down] = np.zeros_like(dos.idos[Spin.up])
        tdos = Dos(self.efermi, dos.energies, dos.tdos)
        idos = Dos(self.efermi, dos.energies, dos.idos)

        return tdos, idos

    def _parse_pdos(self, filpdos):
        """
        Parses the density of states from the output of projwfc.x and/or dos.x

        # TODO: cleanup and rewrite
        """
        pdos = EspressoDos.from_filpdos(filpdos) if filpdos else None

        # The total DOS of NCL-NSOC calcs is missing spin down in QE output
        tdos = pdos._summed_pdos if (self.noncolin and not self.lspinorb) else pdos.tdos
        if self.lspinorb:
            tdos[Spin.down] = np.zeros_like(tdos[Spin.up])
        tdos = Dos(self.efermi, pdos.energies, tdos)
        atomic_states = pdos.atomic_states

        return tdos, self._vaspify_pdos(atomic_states)

    def _vaspify_pdos(self, atomic_states: list[AtomicState] | None):
        """
        Returns the projected DOS in the same format Vasprun uses
        (i.e., the VASP convention)

        # TODO: cleanup and rewrite
        """
        if atomic_states is None:
            return None

        nsites = self.final_structure.num_sites
        pdos = [defaultdict(dict) for _ in range(nsites)]
        if self.lspinorb:
            # TODO: implement this (Clebsch-Gordan coefficients with atomic_proj.xml)
            warnings.warn(
                "Quantum espresso works in the |LJJz> basis when SOC is enabled "
                "while VASP uses the |LLz> basis. Converting between the two is "
                "not currently implemented. pdos will not be lm-decomposed. ",
                DifferentFromVASPWarning,
            )
            pdos = [
                {
                    Orbital(j): {
                        spin: np.zeroslike(atomic_states[0].pdos[Spin.up])
                        for spin in [Spin.up, Spin.down]
                    }
                    for j in range(9)
                }
                for _ in range(nsites)
            ]

        for s in atomic_states:
            if self.noncolin and not self.lspinorb:
                # Noncolinear without SOC
                spin = Spin.up if s.s_z == 0.5 else Spin.down
                pdos[s.site.atom_i - 1][s.orbital][spin] = s.pdos[Spin.up]
            elif not self.noncolin and not self.lspinorb:
                # Colinear spin-polarized and non-spin-polarized
                for spin in s.pdos.keys():
                    pdos[s.site.atom_i - 1][s.orbital][spin] = s.pdos[spin]
            else:
                # Spin orbit case
                orbital = projwfc_orbital_to_vasp(s.l, 2 * s.l + 1)
                pdos[s.site.atom_i - 1][orbital][Spin.up] += s.pdos[Spin.up]

        return pdos

    @staticmethod
    def _parse_vbm_cbm_efermi(b_struct):
        efermi = parse_pwvals(b_struct.get("fermi_energy", None))
        if efermi is not None:
            efermi *= Ha_to_eV

        cbm = parse_pwvals(b_struct.get("lowestUnoccupiedLevel", None))
        if cbm is not None:
            cbm *= Ha_to_eV

        vbm = parse_pwvals(b_struct.get("highestOccupiedLevel", None))
        if vbm is not None:
            vbm *= Ha_to_eV

        return efermi, cbm, vbm

    def _parse_relaxation(
        self, ionic_step_skip, ionic_step_offset, data, output_section
    ):
        calc = self.parameters["control_variables"]["calculation"]
        ionic_steps = []

        if calc in ("vc-relax", "relax"):
            steps = data["step"]
            if isinstance(steps, dict):
                steps = [steps]
            ionic_steps = [
                self._parse_calculation(step)
                for step in steps[ionic_step_offset::ionic_step_skip]
            ]
        ionic_steps.append(self._parse_calculation(output_section, final_step=True))

        final_structure = self._parse_structure(output_section["atomic_structure"])

        return final_structure, ionic_steps

    def _set_pw_calc_params(self, data, input_section, b_struct):
        """
        Sets some useful attributes from the PWscf calculation.
        """
        self.parameters = parse_pwvals(input_section)
        self.prefix = input_section["control_variables"]["prefix"]
        self.espresso_version = parse_pwvals(
            data["general_info"]["creator"]["@VERSION"]
        )
        self.nelec = parse_pwvals(b_struct["nelec"])
        self.noncolin = parse_pwvals(input_section["spin"]["noncolin"])
        self.lspinorb = parse_pwvals(input_section["spin"]["spinorbit"])
        self.lsda = parse_pwvals(input_section["spin"]["lsda"])
        self.nk = parse_pwvals(b_struct["nks"])
        self.nbands = parse_pwvals(
            b_struct["nbnd_up"] if self.lsda else b_struct["nbnd"]
        )

    def _fudge_vasp_params(self):
        """
        Fudges some parameters of the Vasprun object to ensure as many publicly accessible attributes are set as possible. For low stakes stuff like setting a Vasp version, etc.
        """
        self.vasp_version = f"Quantum ESPRESSO v{self.espresso_version}"
        self.potcar_symbols = [
            f"{x} {y}"
            for (x, y) in zip(self.pseudo_filenames, self.pseudo_elements, strict=True)
        ]
        self.potcar_spec = self.pseudo_filenames
        self.incar = Incar()
        self.kpoints = Kpoints(comment="Empty KPOINTS object")
        self.parameters["LSORBIT"] = self.lspinorb
        self.dos_has_errors = False
        self.exception_on_bad_xml = None

    @staticmethod
    def _parse_kpoints(output, T, alat):
        """
        Parses k-points from the XML.
        """
        ks_energies = output["band_structure"]["ks_energies"]
        # If a single dict is provided (e.g., for gamma-point calculations)
        if isinstance(ks_energies, dict):
            ks_energies = [ks_energies]

        nk = len(ks_energies)
        k = np.zeros((nk, 3), float)
        k_weights = np.zeros(nk, float)
        for n in range(nk):
            kp = ks_energies[n]
            k[n] = parse_pwvals(kp["k_point"]["#text"])
            k_weights[n] = parse_pwvals(kp["k_point"]["@weight"])
        # Convert to inverse angstrom
        k_cart = k * (2 * np.pi / alat) * (1 / bohr_to_ang)
        # Convert from cartesian to fractional by multiplying by T
        # TODO: change it to something like
        # Make sure data structures are not messed up though
        # k_frac = (S @ k_cart.T).T
        k_frac = [T @ k for k in k_cart]

        return k_frac, k_cart, k_weights

    @staticmethod
    def _parse_eigen(ks_energies, lsda):
        """
        Parses eigenvalues and occupations from the XML.

        Args:
            ks_energies (list | dict): ks energies
            lsda (bool): true if lsda calculation
        """
        # If a single dict is provided (e.g., for gamma-point calculations)
        if isinstance(ks_energies, dict):
            ks_energies = [ks_energies]

        nk = len(ks_energies)
        nbnd = int(ks_energies[0]["eigenvalues"]["@size"])
        eigenvals = np.zeros((nk, nbnd), float)
        occupations = np.zeros((nk, nbnd), float)
        for n in range(nk):
            kp = ks_energies[n]
            eigenvals[n] = parse_pwvals(kp["eigenvalues"]["#text"])
            occupations[n] = parse_pwvals(kp["occupations"]["#text"])
        eigenvals *= Ha_to_eV
        if lsda:
            nbnd_up = nbnd // 2
            eigenvals = {
                Spin.up: np.dstack(
                    (eigenvals[:, 0:nbnd_up], occupations[:, 0:nbnd_up])
                ),
                Spin.down: np.dstack(
                    (eigenvals[:, nbnd_up:], occupations[:, nbnd_up:])
                ),
            }
        else:
            eigenvals = {Spin.up: np.dstack((eigenvals, occupations))}
        return eigenvals

    @staticmethod
    def _parse_structure(a_struct):
        """
        Parses structure from the XML.
        """
        a1 = parse_pwvals(a_struct["cell"]["a1"])
        a2 = parse_pwvals(a_struct["cell"]["a2"])
        a3 = parse_pwvals(a_struct["cell"]["a3"])
        lattice_matrix = np.stack((a1, a2, a3)) * bohr_to_ang
        lattice = Lattice(lattice_matrix)

        # Read atomic structure
        nat = parse_pwvals(a_struct["@nat"])
        species = [None] * nat
        coords = np.zeros((nat, 3), float)
        atom_dict = a_struct["atomic_positions"]["atom"]
        if nat == 1:
            species = [atom_dict["@name"]]
            coords[0] = parse_pwvals(atom_dict["#text"])
        else:
            for i in range(nat):
                species[i] = atom_dict[i]["@name"]
                coords[i] = parse_pwvals(atom_dict[i]["#text"])
        # NOTE: when the species label is, e.g., Fe1 and Fe2, PMG detects this
        # as Fe+ and Fe2+. Using such labels is common in AFM structures.
        # Need a better way of dealing with this.
        if any(re.match(r"[A-Z][a-z]*[1-9]", s) for s in species):
            warnings.warn(
                "Species labels contain numbers, which is common in AFM structures. "
                "This may cause problems with pymatgen's automatic oxidation state determination."
            )

        coords *= bohr_to_ang
        return Structure(lattice, species, coords, coords_are_cartesian=True)

    @staticmethod
    def _parse_atominfo(a_species):
        """
        Parses atomic symbols and pseudopotential filenames from the XML.
        """
        ntyp = parse_pwvals(a_species["@ntyp"])
        atomic_symbols = [None] * ntyp
        pseudo_filenames = [None] * ntyp
        if ntyp == 1:
            atomic_symbols[0] = a_species["species"]["@name"]
            pseudo_filenames[0] = a_species["species"]["pseudo_file"]
        else:
            for i in range(ntyp):
                atomic_symbols[i] = a_species["species"][i]["@name"]
                pseudo_filenames[i] = a_species["species"][i]["pseudo_file"]

        return atomic_symbols, pseudo_filenames

    def _parse_calculation(self, step, final_step=False):
        """
        Parses forces, stress, convergence, etc. from an ionic step in the XML file.
        """
        istep = {"structure": self._parse_structure(step["atomic_structure"])}
        istep["total_energy"] = parse_pwvals(step["total_energy"])
        istep["total_energy"] = {
            k: v * Ha_to_eV for k, v in istep["total_energy"].items()
        }
        if final_step:
            # TODO: units --> convert scf_accuracy from Ha to eV
            istep["scf_conv"] = parse_pwvals(step["convergence_info"]["scf_conv"])
            if "opt_conv" in step["convergence_info"]:
                istep["ionic_conv"] = parse_pwvals(step["convergence_info"]["opt_conv"])
        else:
            istep["scf_conv"] = parse_pwvals(step["scf_conv"])

        # TODO: double check force units
        natoms = istep["structure"].num_sites
        if "forces" in step:
            istep["forces"] = parse_pwvals(step["forces"]["#text"])
            istep["forces"] = np.array(istep["forces"]).reshape((natoms, 3))
            istep["forces"] *= Ha_to_eV / bohr_to_ang
        else:
            istep["forces"] = None

        # TODO: double check units (is Vasprun.xml eV/A3 or kBar?)
        if "stress" in step:
            istep["stress"] = parse_pwvals(step["stress"]["#text"])
            istep["stress"] = np.array(istep["stress"]).reshape((3, 3))
            istep["stress"] *= Ha_to_eV / (bohr_to_ang) ** 3
        else:
            istep["stress"] = None

        return istep


class UnconvergedPWxmlWarning(Warning):
    """
    Warning for unconverged PWscf run from xml file
    """


class PWxmlParserError(Exception):
    """
    Exception class for PWxml parsing.
    """


class InconsistentWithXMLError(Exception):
    """
    Exception class for data from external files that is inconsistent with the XML file.
    """


class DifferentFromVASPWarning(Warning):
    """
    Warning for differences between QE and VASP outputs
    """


class NotImplementedForQE(Warning):
    """
    Warning for differences between QE and VASP outputs
    """


class ZeroTotalEnergyWarning(Warning):
    """
    Warning for zero total energy. Happens with bands/NSCF calcs in QE
    """
