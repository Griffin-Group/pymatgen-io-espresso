"""
Classes for reading/manipulating PWscf xml files.
"""

from __future__ import annotations

import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import DefaultDict, Literal, Any, Dict, List, Union

import numpy as np
from monty.dev import deprecated
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
import xmltodict
import f90nml
import pandas as pd
from tabulate import tabulate

from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure, Species, Site
from pymatgen.core.units import (
    unitized,
    Ha_to_eV,
    Ry_to_eV,
    eV_to_Ha,
    bohr_to_ang,
    ang_to_bohr,
)

from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
    get_reconstructed_band_structure,
)
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.espresso.utils import parse_pwvals, ibrav_to_lattice, projwfc_orbital_to_vasp
from pymatgen.io.espresso.PWscf.pwin import PWin


# TODO: write docstring
class PWxml(MSONable):
    """
    Vastly improved cElementTree-based parser for vasprun.xml files. Uses
    iterparse to support incremental parsing of large files.
    Speedup over Dom is at least 2x for smallish files (~1Mb) to orders of
    magnitude for larger files (~10Mb).

    **VASP results**

    .. attribute:: ionic_steps

        All ionic steps in the run as a list of
        {"structure": structure at end of run,
        "electronic_steps": {All electronic step data in vasprun file},
        "stresses": stress matrix}

    .. attribute:: tdos

        Total dos calculated at the end of run.

    .. attribute:: idos

        Integrated dos calculated at the end of run.

    .. attribute:: pdos

        List of list of PDos objects. Access as pdos[atomindex][orbitalindex]

    .. attribute:: efermi

        Fermi energy

    .. attribute:: eigenvalues

        Available only if parse_eigen=True. Final eigenvalues as a dict of
        {(spin, kpoint index):[[eigenvalue, occu]]}.
        This representation is based on actual ordering in VASP and is meant as
        an intermediate representation to be converted into proper objects. The
        kpoint index is 0-based (unlike the 1-based indexing in VASP).

    .. attribute:: projected_eigenvalues

        Final projected eigenvalues as a dict of {spin: nd-array}. To access
        a particular value, you need to do
        Vasprun.projected_eigenvalues[spin][kpoint index][band index][atom index][orbital_index]
        This representation is based on actual ordering in VASP and is meant as
        an intermediate representation to be converted into proper objects. The
        kpoint, band and atom indices are 0-based (unlike the 1-based indexing
        in VASP).

    .. attribute:: projected_magnetisation

        Final projected magnetisation as a numpy array with the shape (nkpoints, nbands,
        natoms, norbitals, 3). Where the last axis is the contribution in the 3
        Cartesian directions. This attribute is only set if spin-orbit coupling
        (LSORBIT = True) or non-collinear magnetism (LNONCOLLINEAR = True) is turned
        on in the INCAR.

    .. attribute:: other_dielectric

        Dictionary, with the tag comment as key, containing other variants of
        the real and imaginary part of the dielectric constant (e.g., computed
        by RPA) in function of the energy (frequency). Optical properties (e.g.
        absorption coefficient) can be obtained through this.
        The data is given as a tuple of 3 values containing each of them
        the energy, the real part tensor, and the imaginary part tensor
        ([energies],[[real_partxx,real_partyy,real_partzz,real_partxy,
        real_partyz,real_partxz]],[[imag_partxx,imag_partyy,imag_partzz,
        imag_partxy, imag_partyz, imag_partxz]])

    .. attribute:: nionic_steps

        The total number of ionic steps. This number is always equal
        to the total number of steps in the actual run even if
        ionic_step_skip is used.

    .. attribute:: force_constants

        Force constants computed in phonon DFPT run(IBRION = 8).
        The data is a 4D numpy array of shape (natoms, natoms, 3, 3).

    .. attribute:: normalmode_eigenvals

        Normal mode frequencies.
        1D numpy array of size 3*natoms.

    .. attribute:: normalmode_eigenvecs

        Normal mode eigen vectors.
        3D numpy array of shape (3*natoms, natoms, 3).

    .. attribute:: md_data

        Available only for ML MD runs, i.e., INCAR with ML_LMLFF = .TRUE.
        md_data is a list of dict with the following format:

        [
            {
                'energy': {
                    'e_0_energy': -525.07195568,
                    'e_fr_energy': -525.07195568,
                    'e_wo_entrp': -525.07195568,
                    'kinetic': 3.17809233,
                    'lattice kinetic': 0.0,
                    'nosekinetic': 1.323e-05,
                    'nosepot': 0.0,
                    'total': -521.89385012
                    },
                'forces': [[0.17677989, 0.48309874, 1.85806696], ...],
                'structure': Structure object
            }
        ]

    **VASP inputs**

    .. attribute:: incar

        Incar object for parameters specified in INCAR file.

    .. attribute:: parameters

        Incar object with parameters that vasp actually used, including all
        defaults.

    .. attribute:: kpoints

        Kpoints object for KPOINTS specified in run.

    .. attribute:: actual_kpoints

        List of actual kpoints, e.g.,
        [[0.25, 0.125, 0.08333333], [-0.25, 0.125, 0.08333333],
        [0.25, 0.375, 0.08333333], ....]

    .. attribute:: actual_kpoints_weights

        List of kpoint weights, E.g.,
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667, ....]

    .. attribute:: atomic_symbols

        List of atomic symbols, e.g., ["Li", "Fe", "Fe", "P", "P", "P"]

    .. attribute:: potcar_symbols

        List of POTCAR symbols. e.g.,
        ["PAW_PBE Li 17Jan2003", "PAW_PBE Fe 06Sep2000", ..]

    Author: Omar A. Ashour
    """

    # TODO: update docstring
    def __init__(
        self,
        filename,
        ionic_step_skip=1,
        ionic_step_offset=0,
        parse_dos=True,  # Not implemented
        parse_eigen=True,  # Not used
        parse_projected_eigen=False,  # Not implemented
        parse_potcar_file=True,  # Not implemented
        occu_tol=1e-8,
        separate_spins=False,
        exception_on_bad_xml=True,  # Not used
    ):
        """
        Args:
            filename (str): Filename to parse
            ionic_step_skip (int): If ionic_step_skip is a number > 1,
                only every ionic_step_skip ionic steps will be read for
                structure and energies. Unlike Vasprun, the final energy
                will always be the total energy of the scf calculation
                performed after ionic convergence. This isn ot very useful
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
            parse_dos (bool): Whether to parse the dos. Defaults to True. Set
                to False to shave off significant time from the parsing if you
                are not interested in getting those data.
            parse_eigen (bool): Ignored, kept for Vasprun compatibility. Eigenvalues are
                always parsed from PWscf xml files since it isn't particularly expensive.
            # Not implemented, needs update
            parse_projected_eigen (bool): Whether to parse the projected
                eigenvalues and magnetisation. Defaults to False. Set to True to obtain
                projected eigenvalues and magnetisation. **Note that this can take an
                extreme amount of time and memory.** So use this wisely.
            # Not implemented, needs update
            parse_potcar_file (bool/str): Whether to parse the potcar file to read
                the potcar hashes for the potcar_spec attribute. Defaults to True,
                where no hashes will be determined and the potcar_spec dictionaries
                will read {"symbol": ElSymbol, "hash": None}. By Default, looks in
                the same directory as the vasprun.xml, with same extensions as
                 Vasprun.xml. If a string is provided, looks at that filepath.
            # Needs update, vbm and cbm are determined by PWscf
            occu_tol (float): Sets the minimum tol for the determination of the
                vbm and cbm. Usually the default of 1e-8 works well enough,
                but there may be pathological cases.
            # Needs update
            separate_spins (bool): Whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
            exception_on_bad_xml (bool): Ignored, maintained for Vasprun compatibility
        """
        self.filename = filename
        self.ionic_step_skip = ionic_step_skip
        self.ionic_step_offset = ionic_step_offset
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins

        # Maintained for Vasprun compatibility
        self.exception_on_bad_xml = None

        with zopen(filename, "rt") as f:
            self._parse(
                f,
                parse_dos=parse_dos,
                parse_projected_eigen=parse_projected_eigen,
                ionic_step_skip=ionic_step_skip,
                ionic_step_offset=ionic_step_offset,
            )

        if not self.converged:
            msg = f"{filename} is an unconverged VASP run.\n"
            msg += f"Electronic convergence reached: {self.converged_electronic}.\n"
            msg += f"Ionic convergence reached: {self.converged_ionic}."
            warnings.warn(msg, UnconvergedPWxmlWarning)

    def _parse(
        self,
        stream,
        parse_dos,
        parse_projected_eigen,
        ionic_step_skip,
        ionic_step_offset,
    ):
        self.efermi = None
        self.cbm = None  # Not in Vasprun
        self.vbm = None  # Not in Vasprun
        self.projected_eigenvalues = None
        self.projected_magnetisation = None

        self.generator = None
        self.incar = None

        ionic_steps = []
        md_data = []

        data = xmltodict.parse(stream.read())["qes:espresso"]
        self._debug = data

        input = data["input"]
        output = data["output"]
        self.parameters = self._parse_params(input)
        self.initial_structure = self._parse_structure(input["atomic_structure"])
        # TODO: Vasprun's atomic_symbols includes duplicates, this one doesn't
        self.atomic_symbols, self.pseudo_filenames = self._parse_atominfo(input["atomic_species"])

        nionic_steps = 0
        calc = self.parameters["control_variables"]["calculation"]
        if calc in ("vc-relax", "relax"):
            nionic_steps = len(data["step"])
            for n in range(ionic_step_offset, nionic_steps, ionic_step_skip):
                ionic_steps.append(self._parse_calculation(data["step"][n]))
        nionic_steps += 1
        ionic_steps.append(self._parse_calculation(data["output"], final_step=True))
        self.final_structure = self._parse_structure(output["atomic_structure"])
        # nionic_steps here has a slightly different meaning from the Vasprun class
        # VASP will first do an SCF calculation with the input structure, then perform geometry
        # optimization until you hit EDIFFG or NSW, then it's done.
        # QE does the same thing, but it will also do a final SCF calculation with the optimized
        # structure. In reality, converged QE relax/vc-relax calculations take
        # nionic_steps-1 to converge
        self.nionic_steps = nionic_steps
        self.ionic_steps = ionic_steps

        b_struct = output["band_structure"]
        self.efermi = parse_pwvals(b_struct.get("fermi_energy", None))
        if self.efermi is not None:
            self.efermi *= Ha_to_eV
        self.vbm = parse_pwvals(b_struct.get("highestOccupiedLevel", None))
        if self.vbm is not None:
            self.vbm *= Ha_to_eV
        self.cbm = parse_pwvals(b_struct.get("lowestUnoccupiedLevel", None))
        if self.cbm is not None:
            self.cbm *= Ha_to_eV

        ks_energies = b_struct["ks_energies"]
        # Transformation matrix from cartesian to fractional coordinations
        # in reciprocal space
        T = self.final_structure.lattice.reciprocal_lattice.matrix
        T = np.linalg.inv(T).T
        alat = parse_pwvals(output["atomic_structure"]["@alat"])
        self.kpoints_frac, self.kpoints_cart, self.actual_kpoints_weights = self._parse_kpoints(
            output, T, alat
        )
        self.actual_kpoints = self.kpoints_frac
        self.alat = alat

        lsda = parse_pwvals(input["spin"]["lsda"])
        self.eigenvalues = self._parse_eigen(ks_energies, lsda)
        if parse_projected_eigen:
            # TODO: parse projected magnetisation
            self.projected_eigenvalues = self._parse_projected_eigen(parse_projected_eigen)
        # elif parse_dos:
        # self.tdos, self.idos, self.pdos = self._parse_dos(elem)
        # self.efermi = self.tdos.efermi
        # self.dos_has_errors = False

        self.md_data = md_data
        self.pwscf_version = parse_pwvals(data["general_info"]["creator"]["@VERSION"])

        # TODO: move to a validation function
        nelec = parse_pwvals(b_struct["nelec"])
        noncolin = parse_pwvals(input["spin"]["noncolin"])
        if lsda:
            nbnd = parse_pwvals(b_struct["nbnd_up"])
        else:
            nbnd = parse_pwvals(b_struct["nbnd"])
        factor = 1 if noncolin else 2
        if nbnd <= nelec / factor:
            msg = f"Number of bands ({nbnd}) <= number of electrons/{factor} ({nelec / factor:.4f})"
            msg += ". Pymatgen may not work properly (e.g., BSPlotter)."
            warnings.warn(msg)

    @property
    def structures(self):
        """
        Returns:
             List of Structure objects for the structure at each ionic step.
        """
        return [step["structure"] for step in self.ionic_steps]

    @property
    def converged_electronic(self):
        """
        Returns:
            True if electronic step convergence has been reached in the final
            ionic step
        """
        if self.parameters["control_variables"]["calculation"] in ("nscf", "bands"):
            # PWscf considers NSCF calculations unconverged, but we return True
            # to maintain consistency with the Vasprun class
            return True
        return self.ionic_steps[-1]["scf_conv"]["convergence_achieved"]

    @property
    def converged_ionic(self):
        """
        Returns:
            True if ionic step convergence has been reached
        """
        # Check if dict has 'ionic_conv' key
        if "ionic_conv" in self.ionic_steps[-1]:
            return self.ionic_steps[-1]["ionic_conv"]["convergence_achieved"]
        # To maintain consistency with the Vasprun class, we return True
        # if the calculation didn't involve geometric optimization (scf, nscf, ...)
        return True

    @property
    def converged(self):
        """
        Returns:
            True if a relaxation run is converged both ionically and
            electronically.
        """
        return self.converged_electronic and self.converged_ionic

    @property
    @unitized("eV")
    def final_energy(self):
        """
        Final energy from the PWscf run.
        """
        final_istep = self.ionic_steps[-1]
        total_energy = final_istep["total_energy"]["etot"]
        if total_energy == 0:
            warnings.warn("Calculation has zero total energy. Possibly an NSCF or bands run.")
        return total_energy * Ha_to_eV

    # TODO: implement
    @property
    def complete_dos(self):
        """
        A complete dos object which incorporates the total dos and all
        projected dos.
        """
        print("Not implemented yet.")
        # final_struct = self.final_structure
        # pdoss = {final_struct[i]: pdos for i, pdos in enumerate(self.pdos)}
        # return CompleteDos(self.final_structure, self.tdos, pdoss)

    # TODO: implement
    @property
    def complete_dos_normalized(self) -> CompleteDos:
        """
        A CompleteDos object which incorporates the total DOS and all
        projected DOS. Normalized by the volume of the unit cell with
        units of states/eV/unit cell volume.
        """
        print("Not implemented yet.")
        # final_struct = self.final_structure
        # pdoss = {final_struct[i]: pdos for i, pdos in enumerate(self.pdos)}
        # return CompleteDos(self.final_structure, self.tdos, pdoss, normalize=True)

    @property
    def hubbards(self):
        """
        Hubbard U values used if a vasprun is a GGA+U run. {} otherwise.
        """
        # TODO: ensure that this is correct (not sure how QE treats DFT+U)
        # TODO: check if this was changed in QE v7.2
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
        # TODO: check if this was changed in QE v7.2
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

    @property
    def is_hubbard(self) -> bool:
        """
        True if run is a DFT+U run. Identical implementation to the Vasprun class.
        """
        if len(self.hubbards) == 0:
            return False
        return sum(self.hubbards.values()) > 1e-8

    @property
    def is_spin(self) -> bool:
        """
        True if run is spin-polarized.
        """
        return self.parameters["spin"]["lsda"]

    def get_computed_entry(
        self, inc_structure=True, parameters=None, data=None, entry_id: str | None = None
    ):
        """
        Returns a ComputedEntry or ComputedStructureEntry from the PWxml.
        Tried to maintain consistency with Vasprun.get_computed_entry but it won't be perfect.
        Practically identical implementation to the Vasprun class.

        Args:
            inc_structure (bool): Set to True if you want
                ComputedStructureEntries to be returned instead of
                ComputedEntries.
            parameters (list): Input parameters to include. It has to be one of
                the properties supported by the Vasprun object. If
                parameters is None, a default set of parameters that are
                necessary for typical post-processing will be set.
            data (list): Output data to include. Has to be one of the properties supported
            by the PWxml object.
            entry_id (str): Specify an entry id for the ComputedEntry. Defaults to
                "PWxml-{current datetime}"

        Returns:
            ComputedStructureEntry/ComputedEntry
        """
        if entry_id is None:
            entry_id = f"PWxml-{datetime.datetime.now()}"
        param_names = {
            "is_hubbard",
            "hubbards",
            # "potcar_symbols",
            # "potcar_spec",
            "run_type",
        }
        if parameters:
            param_names.update(parameters)
        params = {p: getattr(self, p) for p in param_names}
        data = {p: getattr(self, p) for p in data} if data is not None else {}

        if inc_structure:
            return ComputedStructureEntry(
                self.final_structure,
                self.final_energy,
                parameters=params,
                data=data,
                entry_id=entry_id,
            )
        return ComputedEntry(
            self.final_structure.composition,
            self.final_energy,
            parameters=params,
            data=data,
            entry_id=entry_id,
        )

    # TODO: implement hybrid
    # TODO: check projections work
    def get_band_structure(
        self,
        kpoints_filename: str | None = None,
        efermi: float | Literal["smart"] | None = "smart",
        line_mode: bool = False,
        force_hybrid_mode: bool = False,
    ) -> BandStructureSymmLine | BandStructure:
        # TODO: update docstring
        """Get the band structure as a BandStructure object.

        Args:
            kpoints_filename: Full path of the PWscf input file from which
                the band structure is generated.
                If none is provided, the code will try to intelligently
                determine the appropriate file by substituting the
                filename of the xml (e.g., SiO2.xml -> SiO2.pwi or SiO2.in)
                The latter is the default behavior.
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
            along symmetry lines)

            NSCF (calc='nscf' or 'bands') calculations are accepted for Line-Mode
            with explicit PWscf input file, and 'crystal', 'crystal_b',
            'tpiba' or 'tpiba_b' K_POINTS card.
            The k-points needs to have data on the kpoint label as commentary.
        """
        if not kpoints_filename:
            input_files = [zpath(self.filename.rsplit(".", 1)[0] + ext) for ext in [".in", ".pwi"]]
            for file_in in input_files:
                kpoints_filename = file_in
                if os.path.exists(file_in):
                    break
        if kpoints_filename and not os.path.exists(kpoints_filename) and line_mode is True:
            raise PWxmlParserError(
                "PW input file needed to obtain band structure along symmetry lines."
            )

        if efermi == "smart":
            e_fermi = self.calculate_efermi()
        elif efermi is None:
            e_fermi = self.efermi
        else:
            e_fermi = efermi

        k_card = None
        if kpoints_filename and os.path.exists(kpoints_filename):
            k_card = PWin.from_file(kpoints_filename).k_points
        lattice_new = Lattice(self.final_structure.lattice.reciprocal_lattice.matrix)

        p_eigenvals: defaultdict[Spin, list] = defaultdict(list)
        eigenvals: defaultdict[Spin, list] = defaultdict(list)

        for spin, v in self.eigenvalues.items():
            v = np.swapaxes(v, 0, 1)
            eigenvals[spin] = v[:, :, 0]

            # TODO: check this works when you implement projected_eigenvalues
            if self.projected_eigenvalues:
                peigen = self.projected_eigenvalues[spin]
                # Original axes for self.projected_eigenvalues are kpoints,
                # band, ion, orb.
                # For BS input, we need band, kpoints, orb, ion.
                peigen = np.swapaxes(peigen, 0, 1)  # Swap kpoint and band axes
                peigen = np.swapaxes(peigen, 2, 3)  # Swap ion and orb axes

                p_eigenvals[spin] = peigen

        # TODO: check how hybrid band structs work in QE
        hybrid_band = False
        # if self.parameters.get("LHFCALC", False) or 0.0 in self.actual_kpoints_weights:
        #    hybrid_band = True

        coords_are_cartesian = False
        if k_card is not None:
            if k_card["options"] in ["crystal", "crystal_b", "tpiba", "tpiba_b"]:
                line_mode = True
                coords_are_cartesian = k_card["options"] in ("tpiba", "tpiba_b")
        if coords_are_cartesian:
            kpoints = [np.array(kpt) for kpt in self.kpoints_cart]
        else:
            kpoints = [np.array(kpt) for kpt in self.kpoints_frac]

        if line_mode:
            labels_dict = {}
            # TODO: check how hybrid stuff works in QE
            if hybrid_band or force_hybrid_mode:
                raise PWxmlParserError("Hybrid band structures not yet supported in line mode.")
            kpoints, eigenvals, p_eigenvals, labels_dict = self._vaspify_kpts_bands(
                kpoints, eigenvals, p_eigenvals, k_card, self.alat
            )
            # TODO: implement support for tpiba and tpiba_b
            # (cartesian coordinates)
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

    # TODO: finish this
    @staticmethod
    def _vaspify_kpts_bands(kpoints, eigenvals, p_eigenvals, k_card, alat):
        """
        Helper function to convert kpoints and eigenvalues to the format
        expected by the BandStructureSymmLine class.

        VASP duplicates k-points along symmetry lines, while QE does not.
        For example, if you do a BS calculation along the path
        X - G - X, VASP will do X - more kpts  G - G - more kpts - X, while QE will do
        X - more kpts - G - more kpts - X. This function duplicates stuff so that
        BandStructureSymmLine works properly.
        """
        labels = [kp["label"] for kp in k_card["data"]]
        kpts = np.array([kp["k"] for kp in k_card["data"]])
        if k_card["options"] in ("tpiba", "tpiba_b"):
            factor = (2 * np.pi / alat) * (1 / bohr_to_ang)
            kpts = [kp * factor for kp in kpts]
        nkpts = [kp["weight"] for kp in k_card["data"]]
        if k_card["options"] in ("crystal_b", "tpiba_b"):
            if "" in labels:
                raise Exception(
                    "A band structure along symmetry lines "
                    "requires a label for each kpoint. "
                    "Check your PWscf input file"
                )
        labels_dict = dict(zip(labels, kpts))
        labels_dict.pop("", None)

        # Figure out the indices of the HSPs that require duplication
        if k_card["options"] in ("crystal_b", "tpiba_b"):
            # pw.x doesn't read the weight of the last k-point, it's treated as just 1
            nkpts[-1] = 1
            nkpts.insert(0, 0)
            hsp_idx = np.cumsum(nkpts)
        else:
            hsp_idx = np.where(np.array(labels) != "")[0][:-1]
        # HSPs with consecutive indices occur at discontinuties, they don't need duplication
        # This also takes care of last HSP with *_b options
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
                        eigenvals[spin], idx + i + 1, p_eigenvals[spin][:, idx + i, :, :], axis=1
                    )

        return kpoints, eigenvals, p_eigenvals, labels_dict

    # TODO: add units
    @property
    def eigenvalue_band_properties(self):
        """
        Band properties from the eigenvalues as a tuple,
        (band gap, cbm, vbm, is_band_gap_direct). In the case of separate_spins=True,
        the band gap, cbm, vbm, and is_band_gap_direct are each lists of length 2,
        with index 0 representing the spin-up channel and index 1 representing
        the spin-down channel.

        Identical implementation to the Vasprun class, with addition of checking against
        the PWscf computed VBM and CBM.
        """
        vbm = -float("inf")
        vbm_kpoint = None
        cbm = float("inf")
        cbm_kpoint = None
        vbm_spins = []
        vbm_spins_kpoints = []
        cbm_spins = []
        cbm_spins_kpoints = []
        if self.separate_spins and len(self.eigenvalues) != 2:
            raise ValueError("The separate_spins flag can only be True if nspin = 2 (LSDA)")

        for d in self.eigenvalues.values():
            if self.separate_spins:
                vbm = -float("inf")
                cbm = float("inf")
            for k, val in enumerate(d):
                for eigenval, occu in val:
                    if occu > self.occu_tol and eigenval > vbm:
                        vbm = eigenval
                        vbm_kpoint = k
                    elif occu <= self.occu_tol and eigenval < cbm:
                        cbm = eigenval
                        cbm_kpoint = k
            if self.separate_spins:
                vbm_spins.append(vbm)
                vbm_spins_kpoints.append(vbm_kpoint)
                cbm_spins.append(cbm)
                cbm_spins_kpoints.append(cbm_kpoint)
        if self.separate_spins:
            return (
                [max(cbm_spins[0] - vbm_spins[0], 0), max(cbm_spins[1] - vbm_spins[1], 0)],
                [cbm_spins[0], cbm_spins[1]],
                [vbm_spins[0], vbm_spins[1]],
                [
                    vbm_spins_kpoints[0] == cbm_spins_kpoints[0],
                    vbm_spins_kpoints[1] == cbm_spins_kpoints[1],
                ],
            )

        # TODO: use some approximation with tolerance
        if self.vbm and vbm != self.vbm:
            delta = np.abs(vbm - self.vbm) * 1000
            msg = f"VBM computed by PWscf is different from the one computed by pymatgen."
            msg += f" (delta = {delta} meV)."
            warnings.warn(msg)
        if self.cbm and cbm != self.cbm:
            delta = np.abs(cbm - self.cbm) * 1000
            msg = f"CBM computed by PWscf is different from the one computed by pymatgen."
            msg += f" (delta = {delta} meV). "
            warnings.warn(msg)
        return max(cbm - vbm, 0), cbm, vbm, vbm_kpoint == cbm_kpoint

    def calculate_efermi(self, tol: float = 0.001):
        """
        Calculate the Fermi level
        PWscf returns the Fermi level for all calculations and the cbm and vbm for all insulators.
        These are stored in PWxml.efermi, PWxml.cbm, and PWxml.vbm.
        However, for insulators, the Fermi level is often slightly off from the exact mid-gap value.

        tol does nothing and is only there to maintain consistency with the
        Vasprun class.
        """
        # If vbm and cbm are both undefined (metallic system), return the Fermi level
        # if vbm is defined and cbm isn't, it's usually a sign of an insulator
        # with as many bands as electrons (often nbnd isn't set in input)
        # Such calculations don't work with BSPlotter()
        if self.vbm is None or self.cbm is None:
            return self.efermi
        return (self.vbm + self.cbm) / 2

    def get_trajectory(self):
        """
        This method returns a Trajectory object, which is an alternative
        representation of self.structures into a single object. Forces are
        added to the Trajectory as site properties.

        Identical implementation to the Vasprun class.

        Returns: a Trajectory
        """
        from pymatgen.core.trajectory import Trajectory

        structs = []
        for step in self.ionic_steps:
            struct = step["structure"].copy()
            struct.add_site_property("forces", step["forces"])
            structs.append(struct)
        return Trajectory.from_structures(structs, constant_lattice=False)

    def as_dict(self):
        """
        JSON-serializable dict representation.

        Almost identical implementation to the Vasprun class.
        """
        d = {
            "pwscf_version": self.pwscf_version,
            "has_pwscf_completed": self.converged,
            "nsites": len(self.final_structure),
        }
        comp = self.final_structure.composition
        d["unit_cell_formula"] = comp.as_dict()
        d["reduced_cell_formula"] = Composition(comp.reduced_formula).as_dict()
        d["pretty_formula"] = comp.reduced_formula
        symbols = self.atomic_symbols
        d["is_hubbard"] = self.is_hubbard
        d["hubbards"] = self.hubbards

        unique_symbols = sorted(set(self.atomic_symbols))
        d["elements"] = unique_symbols
        d["nelements"] = len(unique_symbols)

        d["run_type"] = self.run_type

        vin = {
            # TODO: implement this later
            # "incar": dict(self.incar.items()),
            "crystal": self.initial_structure.as_dict(),
            # "kpoints": self.kpoints.as_dict(),
        }
        actual_kpts = [
            {
                "abc": list(self.actual_kpoints[i]),
                "weight": self.actual_kpoints_weights[i],
            }
            for i in range(len(self.actual_kpoints))
        ]
        # vin["kpoints"]["actual_points"] = actual_kpts
        vin["nkpoints"] = len(actual_kpts)
        vin["pseudo_filenames"] = self.pseudo_filenames
        # vin["potcar_spec"] = self.potcar_spec
        # vin["potcar_type"] = [s.split(" ")[0] for s in self.potcar_symbols]
        vin["parameters"] = dict(self.parameters.items())
        vin["lattice_rec"] = self.final_structure.lattice.reciprocal_lattice.as_dict()
        d["input"] = vin

        nsites = len(self.final_structure)

        try:
            vout = {
                "ionic_steps": self.ionic_steps,
                "final_energy": self.final_energy,
                "final_energy_per_atom": self.final_energy / nsites,
                "crystal": self.final_structure.as_dict(),
                "efermi": self.efermi,
            }
        except (ArithmeticError, TypeError):
            vout = {
                "ionic_steps": self.ionic_steps,
                "final_energy": self.final_energy,
                "final_energy_per_atom": None,
                "crystal": self.final_structure.as_dict(),
                "efermi": self.efermi,
            }

        if self.eigenvalues:
            eigen = {str(spin): v.tolist() for spin, v in self.eigenvalues.items()}
            vout["eigenvalues"] = eigen
            (gap, cbm, vbm, is_direct) = self.eigenvalue_band_properties
            vout.update({"bandgap": gap, "cbm": cbm, "vbm": vbm, "is_gap_direct": is_direct})

            if self.projected_eigenvalues:
                vout["projected_eigenvalues"] = {
                    str(spin): v.tolist() for spin, v in self.projected_eigenvalues.items()
                }

            if self.projected_magnetisation is not None:
                vout["projected_magnetisation"] = self.projected_magnetisation.tolist()

        d["output"] = vout
        return jsanitize(d, strict=True)

    @staticmethod
    def _parse_params(params):
        # TODO: implement this into some input file object
        return parse_pwvals(params)

    # TODO: implement
    @staticmethod
    def _parse_projected_eigen(filename):
        print("Not implemented.")

    @staticmethod
    def _parse_kpoints(output, T, alat):
        ks_energies = output["band_structure"]["ks_energies"]

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
        k_frac = [T @ k for k in k_cart]

        return k_frac, k_cart, k_weights

    @staticmethod
    def _parse_eigen(ks_energies, lsda):
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
                Spin.up: np.dstack((eigenvals[:, 0:nbnd_up], occupations[:, 0:nbnd_up])),
                Spin.down: np.dstack((eigenvals[:, nbnd_up:], occupations[:, nbnd_up:])),
            }
        else:
            eigenvals = {Spin.up: np.dstack((eigenvals, occupations))}
        return eigenvals

    @staticmethod
    def _parse_structure(a_struct):
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
        # Need to think about how to deal with this.
        if any([re.match(r"[A-Z][a-z]*[1-9]", s) for s in species]):
            warnings.warn(
                "Species labels contain numbers, which is common in AFM structures. "
                "This may cause problems with pymatgen's automatic oxidation state determination."
            )

        coords *= bohr_to_ang
        return Structure(lattice, species, coords, coords_are_cartesian=True)

    @staticmethod
    def _parse_atominfo(a_species):
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
        istep = {}
        istep["structure"] = self._parse_structure(step["atomic_structure"])

        istep["total_energy"] = parse_pwvals(step["total_energy"])
        istep["total_energy"] = {k: v * Ha_to_eV for k, v in istep["total_energy"].items()}
        if final_step:
            # TODO: units --> convert scf_accuracy from Ha to eV
            istep["scf_conv"] = parse_pwvals(step["convergence_info"]["scf_conv"])
            if "opt_conv" in step["convergence_info"]:
                istep["ionic_conv"] = parse_pwvals(step["convergence_info"]["opt_conv"])
        else:
            istep["scf_conv"] = parse_pwvals(step["scf_conv"])

        # TODO: parse stress from last step
        # TODO: force units
        natoms = istep["structure"].num_sites
        if "forces" in step:
            istep["forces"] = parse_pwvals(step["forces"]["#text"])
            istep["forces"] = np.array(istep["forces"]).reshape((natoms, 3))
            istep["forces"] *= Ha_to_eV / bohr_to_ang
        else:
            istep["forces"] = None

        return istep


class UnconvergedPWxmlWarning(Warning):
    """
    Warning for unconverged PWscf run.
    """


class PWxmlParserError(Exception):
    """
    Exception class for PWscf parsing.
    """
