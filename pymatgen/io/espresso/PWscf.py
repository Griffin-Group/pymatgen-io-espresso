"""
Classes for reading/manipulating/writing PWScf input and output files.
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
from typing import DefaultDict, Literal

import numpy as np
from monty.dev import deprecated
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
import xmltodict

from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.units import unitized
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
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri


def _parse_pwvals(val):
    """
    Helper method to parse values in the PWscf xml files. Supports array, dict, bool, float and int.

    Returns original string (or list of substrings) if no match is found.
    """
    # regex to match floats but not integers, including scientific notation
    float_regex = "[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?"
    # regex to match just integers (signed or unsigned)
    int_regex = "^(\+|-)?\d+$"
    if type(val) == dict:
        return {k: _parse_pwvals(v) for k, v in val.items()}
    if type(val) == list:
        return [_parse_pwvals(x) for x in val]
    if val is None:
        return None
    if " " in val:
        return [_parse_pwvals(x) for x in val.split()]
    if val in ("true", "false"):
        return val == "true"
    if re.fullmatch(float_regex, val):
        return float(val)
    if re.fullmatch(int_regex, val):
        return int(val)
    return val


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

    Author: Shyue Ping Ong
    """

    def __init__(
        self,
        filename,
        ionic_step_skip=1,
        ionic_step_offset=0,
        parse_dos=True,
        parse_eigen=True,
        parse_projected_eigen=False,
        parse_potcar_file=True,
        occu_tol=1e-8,
        separate_spins=False,
        exception_on_bad_xml=True,
    ):
        """
        Args:
            filename (str): Filename to parse
            ionic_step_skip (int): If ionic_step_skip is a number > 1,
                only every ionic_step_skip ionic steps will be read for
                structure and energies. This is very useful if you are parsing
                very large vasprun.xml files and you are not interested in every
                single ionic step. Note that the final energies may not be the
                actual final energy in the vasprun.
            ionic_step_offset (int): Used together with ionic_step_skip. If set,
                the first ionic step read will be offset by the amount of
                ionic_step_offset. For example, if you want to start reading
                every 10th structure but only from the 3rd structure onwards,
                set ionic_step_skip to 10 and ionic_step_offset to 3. Main use
                case is when doing statistical structure analysis with
                extremely long time scale multiple VASP calculations of
                varying numbers of steps.
            parse_dos (bool): Whether to parse the dos. Defaults to True. Set
                to False to shave off significant time from the parsing if you
                are not interested in getting those data.
            parse_eigen (bool): Whether to parse the eigenvalues. Defaults to
                True. Set to False to shave off significant time from the
                parsing if you are not interested in getting those data.
            parse_projected_eigen (bool): Whether to parse the projected
                eigenvalues and magnetisation. Defaults to False. Set to True to obtain
                projected eigenvalues and magnetisation. **Note that this can take an
                extreme amount of time and memory.** So use this wisely.
            parse_potcar_file (bool/str): Whether to parse the potcar file to read
                the potcar hashes for the potcar_spec attribute. Defaults to True,
                where no hashes will be determined and the potcar_spec dictionaries
                will read {"symbol": ElSymbol, "hash": None}. By Default, looks in
                the same directory as the vasprun.xml, with same extensions as
                 Vasprun.xml. If a string is provided, looks at that filepath.
            occu_tol (float): Sets the minimum tol for the determination of the
                vbm and cbm. Usually the default of 1e-8 works well enough,
                but there may be pathological cases.
            separate_spins (bool): Whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
            exception_on_bad_xml (bool): Whether to throw a ParseException if a
                malformed XML is detected. Default to True, which ensures only
                proper vasprun.xml are parsed. You can set to False if you want
                partial results (e.g., if you are monitoring a calculation during a
                run), but use the results with care. A warning is issued.
        """
        self.filename = filename
        self.ionic_step_skip = ionic_step_skip
        self.ionic_step_offset = ionic_step_offset
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins
        self.exception_on_bad_xml = exception_on_bad_xml

        with zopen(filename, "rt") as f:
            self._parse(
                f,
                parse_dos=parse_dos,
                parse_eigen=parse_eigen,
                parse_projected_eigen=parse_projected_eigen,
                ionic_step_skip=ionic_step_skip,
                ionic_step_offset=ionic_step_offset,
            )

        if not self.converged:
            msg = f"{filename} is an unconverged VASP run.\n"
            msg += f"Electronic convergence reached: {self.converged_electronic}.\n"
            msg += f"Ionic convergence reached: {self.converged_ionic}."
            warnings.warn(msg, UnconvergedPWscfWarning)

    def _parse(self, stream, parse_dos, parse_eigen, parse_projected_eigen, ionic_step_skip, ionic_step_offset):
        self.efermi = None
        self.cbm = None  # Not in Vasprun
        self.vbm = None  # Not in Vasprun
        self.eigenvalues = None
        self.projected_eigenvalues = None
        self.projected_magnetisation = None

        self.generator = None
        self.incar = None

        ionic_steps = []
        md_data = []

        data = xmltodict.parse(stream.read())["qes:espresso"]

        input = data["input"]
        output = data["output"]
        self.parameters = self._parse_params(input)
        self.initial_structure = self._parse_structure(input["atomic_structure"])
        self.atomic_symbols, self.pseudo_filenames = self._parse_atominfo(input["atomic_species"])

        nionic_steps = 0
        calc = self.parameters["control_variables"]["calculation"]
        if calc in ("vc-relax", "relax"):
            nionic_steps = len(data["step"])
            for n in range(ionic_step_offset, nionic_steps, ionic_step_skip):
                ionic_steps.append(self._parse_calculation(data["step"][n]))
        nionic_steps += 1
        ionic_steps.append(self._parse_calculation(data["output"], final_calc=True))
        # nionic_steps here has a slightly different meaning from the Vasprun class
        # VASP will first do an SCF calculation with the input structure, then perform geometry
        # optimization until you hit EDIFFG or NSW, then it's done.
        # QE does the same thing, but it will also do a final SCF calculation with the optimized
        # structure. In reality, converged QE relax/vc-relax calculations take nionic_steps-1 to converge
        self.nionic_steps = nionic_steps
        self.ionic_steps = ionic_steps

        b_struct = output["band_structure"]
        self.efermi = _parse_pwvals(b_struct.get("fermi_energy", None))
        self.vbm = _parse_pwvals(b_struct.get("highestOccupiedLevel", None))
        self.cbm = _parse_pwvals(b_struct.get("lowestUnoccupiedLevel", None))

        ks_energies = b_struct["ks_energies"]
        self.actual_kpoints, self.actual_kpoints_weights = self._parse_kpoints(ks_energies)
        lsda = _parse_pwvals(input["spin"]["lsda"])
        if parse_eigen:
            self.eigenvalues = self._parse_eigen(ks_energies, lsda)
        ##elif parse_projected_eigen and tag == "projected":
        # self.projected_eigenvalues, self.projected_magnetisation = self._parse_projected_eigen(elem)
        ##elif parse_dos and tag == "dos":
        # self.tdos, self.idos, self.pdos = self._parse_dos(elem)
        # self.efermi = self.tdos.efermi
        # self.dos_has_errors = False

        self.final_structure = self._parse_structure(output["atomic_structure"])
        self.md_data = md_data
        self.pwscf_version = _parse_pwvals(data["general_info"]["creator"]["@VERSION"])

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
        else:
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
        else:
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
    # @unitized("eV")
    # TODO: add units
    def final_energy(self):
        """
        Final energy from the PWscf run.
        """
        final_istep = self.ionic_steps[-1]
        total_energy = final_istep["total_energy"]["etot"]
        if total_energy == 0:
            warnings.warn("Calculation has zero total energy. Possibly an NSCF or bands run.")
        return total_energy

    # TODO: add units
    def calculate_efermi(self, tol: float = 0.001):
        """
        Calculate the Fermi level using a robust algorithm.
        PWscf returns the Fermi level for all calculations and the cbm and vbm for all insulators.
        These are stored in PWxml.efermi, PWxml.cbm, and PWxml.vbm.
        However, for insulators, the Fermi level is often slightly off from the exact mid-gap value.

        tol does nothing and is only there to maintain consistency with the
        Vasprun class.
        """
        # If vbm and cbm are undefined (metallic system), return the Fermi level
        if self.vbm is None and self.cbm is None:
            return self.efermi
        else:
            return (self.vbm + self.cbm) / 2
        
        #all_eigs = np.concatenate([eigs[:, :, 0].transpose(1, 0) for eigs in self.eigenvalues.values()])
        #vbm = np.max(all_eigs[all_eigs < self.efermi])
        #cbm = np.min(all_eigs[all_eigs > self.efermi])

        ## TODO: proper unit handling
        ## TODO: use some approximation with tolerance
        #if vbm != self.vbm:
        #    delta = np.abs(vbm - self.vbm)*27.2*1000 
        #    msg = f"VBM computed by PWscf is different from the one computed by pymatgen (delta = {delta} meV). "
        #    msg += "Using the one computed by Pymatgen to compute fermi level."
        #    warnings.warn(msg)
        #if cbm != self.cbm:
        #    delta = np.abs(cbm - self.cbm)*27.2*1000 # TODO: proper unit handling
        #    msg = f"CBM computed by PWscf is different from the one computed by pymatgen (delta = {delta} meV). "
        #    msg += "Using the one computed by Pymatgen to compute fermi level."
        #    warnings.warn(msg)

    @staticmethod
    def _parse_params(params):
        # TODO: implement this into some input file object
        return _parse_pwvals(params)

    @staticmethod
    def _parse_kpoints(ks_energies):
        nk = len(ks_energies)
        k = np.zeros((nk, 3), float)
        k_weights = np.zeros(nk, float)
        for n in range(nk):
            kp = ks_energies[n]
            k[n] = _parse_pwvals(kp["k_point"]["#text"])
            k_weights[n] = _parse_pwvals(kp["k_point"]["@weight"])
        return k, k_weights

    @staticmethod
    def _parse_eigen(ks_energies, lsda):
        nk = len(ks_energies)
        nbnd = int(ks_energies[0]["eigenvalues"]["@size"])
        eigenvals = np.zeros((nk, nbnd), float)
        occupations = np.zeros((nk, nbnd), float)
        for n in range(nk):
            kp = ks_energies[n]
            eigenvals[n] = _parse_pwvals(kp["eigenvalues"]["#text"])
            occupations[n] = _parse_pwvals(kp["occupations"]["#text"])
        # TODO: energy units
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
        # TODO: deal with units
        a1 = _parse_pwvals(a_struct["cell"]["a1"])
        a2 = _parse_pwvals(a_struct["cell"]["a2"])
        a3 = _parse_pwvals(a_struct["cell"]["a3"])
        lattice_matrix = np.stack((a1, a2, a3))
        lattice = Lattice(lattice_matrix)

        # Read atomic structure
        nat = _parse_pwvals(a_struct["@nat"])
        species = [None] * nat
        coords = np.zeros((nat, 3), float)
        atom_dict = a_struct["atomic_positions"]["atom"]
        if nat == 1:
            species = [atom_dict["@name"]]
            coords[0] = _parse_pwvals(atom_dict["#text"])
        else:
            for i in range(nat):
                species[i] = atom_dict[i]["@name"]
                coords[i] = _parse_pwvals(atom_dict[i]["#text"])

        return Structure(lattice, species, coords, coords_are_cartesian=True)

    @staticmethod
    def _parse_atominfo(a_species):
        ntyp = _parse_pwvals(a_species["@ntyp"])
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

    def _parse_calculation(self, step, final_calc=False):
        istep = {}
        istep["structure"] = self._parse_structure(step["atomic_structure"])

        # TODO: energy units
        istep["total_energy"] = _parse_pwvals(step["total_energy"])
        if final_calc:
            istep["scf_conv"] = _parse_pwvals(step["convergence_info"]["scf_conv"])
            if "opt_conv" in step["convergence_info"]:
                istep["ionic_conv"] = _parse_pwvals(step["convergence_info"]["opt_conv"])
        else:
            istep["scf_conv"] = _parse_pwvals(step["scf_conv"])

        natoms = istep["structure"].num_sites
        if "forces" in step:
            istep["forces"] = _parse_pwvals(step["forces"]["#text"])
            istep["forces"] = np.array(istep["forces"]).reshape((natoms, 3))
        else:
            istep["forces"] = None

        return istep


class UnconvergedPWscfWarning(Warning):
    """
    Warning for unconverged PWscf run.
    """
