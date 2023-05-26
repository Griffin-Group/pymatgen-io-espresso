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
    Helper method to parse values in the PWscf xml files. Supports array, bool, float and int.

    It is assumed that it won't be passed an actual string
    """
    # regex to match floats but not integers
    regex='[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?'

    if ' ' in val:
        return [_parse_pwvals(x) for x in val.split()]
    if val in ("true", "false"):
        return True if val == "true" else False
    if re.search(regex, val):
        return float(val)
    return int(val)

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
        ionic_step_skip=None,
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
            if ionic_step_skip or ionic_step_offset:
                # remove parts of the xml file and parse the string
                run = f.read()
                steps = run.split("<calculation>")
                # The text before the first <calculation> is the preamble!
                preamble = steps.pop(0)
                self.nionic_steps = len(steps)
                new_steps = steps[ionic_step_offset :: int(ionic_step_skip)]
                # add the tailing information in the last step from the run
                to_parse = "<calculation>".join(new_steps)
                if steps[-1] != new_steps[-1]:
                    to_parse = f"{preamble}<calculation>{to_parse}{steps[-1].split('</calculation>')[-1]}"
                else:
                    to_parse = f"{preamble}<calculation>{to_parse}"
                self._parse(
                    StringIO(to_parse),
                    parse_dos=parse_dos,
                    parse_eigen=parse_eigen,
                    parse_projected_eigen=parse_projected_eigen,
                )
            else:
                self._parse(
                    f,
                    parse_dos=parse_dos,
                    parse_eigen=parse_eigen,
                    parse_projected_eigen=parse_projected_eigen,
                )
                self.nionic_steps = len(self.ionic_steps)

        if (
            self.incar.get("ALGO", "") not in ["CHI", "BSE"]
            and (not self.converged)
            and self.parameters.get("IBRION", -1) != 0
        ):
            msg = f"{filename} is an unconverged VASP run.\n"
            msg += f"Electronic convergence reached: {self.converged_electronic}.\n"
            msg += f"Ionic convergence reached: {self.converged_ionic}."
            warnings.warn(msg, UnconvergedVASPWarning)

    def _parse(self, stream, parse_dos, parse_eigen, parse_projected_eigen):

        self.efermi = None
        self.eigenvalues = None
        self.projected_eigenvalues = None
        self.projected_magnetisation = None
        self.dielectric_data = {}
        self.other_dielectric = {}
        self.incar = {}
        ionic_steps = []

        md_data = []

        data = xmltodict.parse(stream.read())['qes:espresso']
        
        self.generator = None
        self.incar = None

        self.actual_kpoints, self.actual_kpoints_weights, self.eigenvals = self._parse_kpoints(data['output']['band_structure']['ks_energies'])
        #self.parameters = self._parse_params(data['input'])
        #self.initial_structure = self._parse_structure(elem)
        #self.atomic_symbols, self.potcar_symbols = self._parse_atominfo(elem)
        #self.potcar_spec = [{"titel": p, "hash": None} for p in self.potcar_symbols]
        #ionic_steps.append(self._parse_calculation(elem))
        ##elif parse_dos and tag == "dos":
        #self.tdos, self.idos, self.pdos = self._parse_dos(elem)
        #self.efermi = self.tdos.efermi
        #self.dos_has_errors = False
        ##elif parse_eigen and tag == "eigenvalues":
        #self.eigenvalues = self._parse_eigen(elem)
        ##elif parse_projected_eigen and tag == "projected":
        #self.projected_eigenvalues, self.projected_magnetisation = self._parse_projected_eigen(elem)
        #self.dielectric_data["density"] = self._parse_diel(elem)
        #self.dielectric_data["velocity"] = self._parse_diel(elem)
        #self.dielectric_data["density"] = self._parse_diel(elem)
        #self.dielectric_data["velocity"] = self._parse_diel(elem)
        #self.other_dielectric[comment] = self._parse_diel(elem)

        #self.optical_transition = np.array(_parse_varray(elem))
        #self.final_structure = self._parse_structure(elem)
        #hessian, eigenvalues, eigenvectors = self._parse_dynmat(elem)
        #natoms = len(self.atomic_symbols)
        #hessian = np.array(hessian)
        #self.force_constants = np.zeros((natoms, natoms, 3, 3), dtype="double")
        #self.force_constants[i, j] = hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
        #self.normalmode_eigenvals = np.array(eigenvalues)
        #self.normalmode_eigenvecs = np.array(phonon_eigenvectors)
        #self.ionic_steps = ionic_steps
        #self.md_data = md_data
        #self.pwscf_version = 

    @staticmethod
    def _parse_kpoints(ks_energies):
        nk = len(ks_energies)
        #nbnd = int(ks_energies[0]["eigenvalues"]["@size"])
        k = np.zeros((nk, 3), float)
        k_weights = np.zeros(nk, float)
        #eigenvals = np.zeros((nk, nbnd), float)
        #occupations = np.zeros((nk, nbnd), float)
        for n in range(nk):
            kp = ks_energies[n]
            k[n] = _parse_pwvals(kp["k_point"]["#text"])
            k_weights[n] = _parse_pwvals(kp["k_point"]["@weight"])
            #eigenvals[n] = _parse_pwvals(kp["eigenvalues"]["#text"])
            #occupations[n] = _parse_pwvals(kp["occupations"]["#text"])

        return k, k_weights#, eigenvals, occupations
