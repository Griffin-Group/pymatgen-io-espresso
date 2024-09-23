"""
Classes for reading/manipulating Projwfc.x files.
"""

import itertools
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import pandas as pd
from monty.json import MSONable
from tabulate import tabulate

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Site, Structure
from pymatgen.core.units import (
    Ry_to_eV,
    bohr_to_ang,
)
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.io.espresso.utils import (
    ibrav_to_lattice,
    parse_pwvals,
    projwfc_orbital_to_vasp,
)


class Projwfc(MSONable):
    """
    Class to parse projwfc.x output. Supports parsing from projwfc.out (projwfc.x's
    stdout), filproj, and atomic_proj.xml files. filproj is recommended for parsing, as
    it is the most complete source of data. See
    [this page](../../../../../../dev_notes/projwfc_output_comparison) for a comparison of the
    three files.

    Attributes:
        parameters (dict): Parameters parsed from the header of the file.
            Contents depend on the source of the data.
        structure (Structure): Structure object parsed from the file.
        lspinorb (bool | None): Whether the calculation includes spin-orbit coupling.
        noncolin (bool | None): Whether the calculation is noncolinear.
        lsda (bool | None): Whether the calculation is spin-polarized.
        nstates (int): Number of atomic states parsed from the file.
        atomic_states (list[AtomicState]): List of AtomicState objects parsed from
            the file. Ordered in the same way projwfc.x orders them.
        nk (int): Number of k-points parsed from the file.
        nbands (int): Number of bands parsed from the file.
        k (np.ndarray | None): k-points parsed from the file. Shape is (nkstot, 3).
        k_weights (np.ndarray | None): k-point weights parsed from the file.
            Shape is (nkstot,).
        eigenvals (np.ndarray | None): Eigenvalues parsed from the file.
            Shape is (nkstot, nbnd).
        proj_source (str): Source of the data. One of "projwfc.out", "filproj",
            or "atomic_proj.xml".
    """

    def __init__(
        self,
        parameters: dict[str, any],
        filename: str | os.PathLike,
        proj_source: str,
        structure: Structure | None = None,
        atomic_states: list["AtomicState"] | None = None,
        k: np.ndarray[float] | None = None,
        k_weights: np.ndarray[float] | None = None,
        eigenvals: np.ndarray[float] | None = None,
    ):
        """
        Constructor for Projwfc object. Shouldn't really be used directly, use one of the class methods instead: `from_projwfcout`, `from_filproj`, or `from_xml`.

        A lot of arguments and parameters can be none since not all the files contain
        the same information. The parameters dictionary is parsed from the header of the
        file and can contain some or all of the following keys:
            - natomwfc: Number of atomic states
            - nr1x, nr2x, nr3x: Number of grid points on the coarse grid
            - nr1, nr2, nr3: Number of grid points on the fine grid
            - gcutm: plane wave cutoff as a g-vector
            - dual: ratio between charge density and plane wave cutoffs
            - nkstot: Number of k-points
            - nbnd: Number of bands
            - nine: Always the number 9
            - lsda: Whether the calculation is spin-polarized
            - lspinorb: Whether the calculation includes spin-orbit coupling
            - noncolin: Whether the calculation is noncolinear

        Args:
        parameters (dict): Parameters parsed from the header of the file.
            Contents depend on the source of the data.
        filename (str | os.PathLike): Path to the file
        proj_source (str): Source of the data. One of "projwfc.out", "filproj",
            or "atomic_proj.xml".
        structure (Structure): Structure object parsed from the file.
        atomic_states (list[AtomicState]): List of AtomicState objects parsed from
            the file. Ordered in the same way projwfc.x orders them.
        k (np.ndarray): k-points parsed from the file. Shape is (nkstot, 3).
        k_weights (np.ndarray): k-point weights parsed from the file.
            Shape is (nkstot,).
        eigenvals (np.ndarray): Eigenvalues parsed from the file.
            Shape is (nkstot, nbnd).

        """
        self.parameters = parameters
        self.structure = structure
        self.lspinorb = parameters.get("lspinorb")
        self.noncolin = parameters.get("noncolin")
        self.lsda = parameters.get("lsda")
        self.nstates = parameters["natomwfc"]
        self.atomic_states = [] if atomic_states is None else atomic_states
        self.nk = parameters["nkstot"]
        self.nbands = parameters["nbnd"]
        self.k = [] if k is None else k
        self.k_weights = [] if k_weights is None else k_weights
        self.eigenvals = {} if eigenvals is None else eigenvals
        self.proj_source = proj_source
        self._filename = filename

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """
        Equality test. Meant for checking that the two objects come from
        the same calculation, not that they are identical. It also assumes that the atomic states are ordered identically, and it checks them against each other using the `AtomicState.__eq__` method (see that method for a description of how the comparison is done).

        This dunder method is really only intended as a check before adding two Projwfc objects together. See the `Projwfc.__add__` method for more information.
        """
        if not isinstance(other, Projwfc):
            return False

        # Not all sources of Projwfc data will have a structure
        same_structure = (
            self.structure == other.structure
            if (self.structure and other.structure)
            else True
        )
        same_states = all(
            s1 == s2
            for s1, s2 in zip(self.atomic_states, other.atomic_states, strict=False)
        )
        return all(
            [
                same_structure,
                same_states,
                self.lspinorb == other.lspinorb,
                self.noncolin == other.noncolin,
                self.nstates == other.nstates,
                self.nk == other.nk,
                self.nbands == other.nbands,
            ]
        )

    def __add__(self, other: "Projwfc"):
        """
        Combine two Projwfc objects. This is intended for combining one object with the
        spin up channel and another with the spin down. This is only ever necessary when
        parsing filproj for a spin-polarized calculation, since the two channels are
        stored in separate files.

        Before addition, this method checks that the two objects must come from
        the same calculation see the `Projwfc.__eq__` method for more information.
        This check is guaranteed to pass if the two objects are parsed from two filproj
        files produced by the same calculation. Returns a new Projwfc object with the
        combined data.
        """
        if not isinstance(other, Projwfc):
            raise ValueError("Can only add Projwfc objects to other Projwfc objects.")
        if self != other:
            raise InconsistentProjwfcDataError(
                "Can only add Projwfc objects from the same calculation."
            )

        # Check that one is spin up and the other is spin down
        # Get all the spins of each object. These are the keys of the projection
        # attribute.
        spin1 = {
            spin for state in self.atomic_states for spin in state.projections.keys()
        }
        spin2 = {
            spin for state in other.atomic_states for spin in state.projections.keys()
        }
        if len(spin1) != 1 or len(spin2) != 1:
            raise InconsistentProjwfcDataError(
                (
                    "You are trying to add two Projwfc objects with multiple spins. "
                    "This should only be used to add objects with one spin each."
                )
            )
        spin1, spin2 = spin1.pop(), spin2.pop()
        if spin1 == spin2:
            raise InconsistentProjwfcDataError(
                "Can only add Projwfc objects with opposite spins."
            )

        result = deepcopy(self)
        for s1, s2 in zip(result.atomic_states, other.atomic_states, strict=True):
            s1.projections |= s2.projections

        return result

    def __str__(self):
        """
        String representation of the object. This is intended to be a human-readable
        summary of the data parsed from the file. It includes the header information,
        a summary of the structure, and a summary of the atomic states. It also
        includes information about which data was parsed and which was not.
        """

        # Incompletely parsed calculations (xml) won't have the noncolin or lspinorb
        if self.noncolin is None:
            header = "Unknown "
        elif self.lspinorb:
            header = "Spin-orbit "
        elif self.noncolin:
            header = "Noncolinear "
        else:
            header = "Colinear "
        if self.lsda:
            header += "(spin-polarized) "
        header += f"calculation with {self.nk} k-points and {self.nbands} bands."
        out = [header, f"Filename: {self._filename}"]
        k_parsed = f"K-points parsed: {np.any(self.k)} "
        if np.any(self.k):
            k_parsed += f"(Units: {self.parameters['k_unit']})"
        out.extend(
            (
                k_parsed,
                f"K-point weights parsed: {np.any(self.k_weights)}",
                f"Eigenvalues parsed: {bool(self.eigenvals)}",
                f"Projections data source: {self.proj_source}",
                "\n------------ Structure ------------",
            )
        )
        if self.structure:
            out.extend(str(self.structure).split("\n")[:5])
            out.append(f"Sites ({self.structure.num_sites})")
            # Almost identical to Structure.__str__
            data = []
            for site in self.structure.sites:
                row = [
                    site.atom_i,
                    site.species_string,
                    *[f"{j:0.6f}" for j in site.frac_coords],
                    site.Z,
                ]
                data.append(row)
            out.append(
                tabulate(
                    data,
                    headers=["#", "SP", "a", "b", "c", "Z val."],
                )
            )
        else:
            out.append("Structure not parsed.")
        out.append("\n---------- Atomic States ----------")

        if self.atomic_states and self.atomic_states[0].l is not None:
            data = []
            headers = ["State #", "SP (#)", "Orbital", "l"]
            if self.lspinorb:
                headers.extend(["j", "mj"])
            elif self.noncolin:
                headers.extend(["m", "s_z"])
            else:
                headers.extend(["m"])

            for state in self.atomic_states:
                _, orbital_str = state._to_projwfc_state_string()
                orb = orbital_str.split()[1]
                atom = state.site.species_string + " (" + str(state.site.atom_i) + ")"
                row = [state.state_i, atom, orb, state.l]
                if state.j:
                    row.extend([state.j, state.mj])
                elif state.s_z:
                    row.extend([state.m, state.s_z])
                else:
                    row.extend([state.m])
                data.append(row)

            out.append(tabulate(data, headers=headers))
        else:
            out.append(
                f"Found {self.nstates} atomic states, but their type is unknown."
            )

        return "\n".join(out)

    @classmethod
    def from_projwfcout(
        cls, filename: str | os.PathLike, parse_projections: bool = True
    ):
        """
        Initialize from a projwfc.out file (stdout of projwfc.x)

        Args:
            filename (str | os.PathLike): Path to the file
            parse_projections (bool): Whether to parse the projections. If False, only
                the header of the file is parsed.
        """

        with open(filename, "r") as f:
            if parse_projections:
                data = f.read()
            else:
                # TODO: Does it matter how many lines you read if you don't parse
                # the projections? Need benchmarking
                nlines = 1000
                head = list(itertools.islice(f, nlines))
                data = "\n".join(head)

        parameters, atomic_states = cls._parse_projwfcout_header(data)
        k, eigenvals = [], {}
        if parse_projections:
            k, eigenvals, atomic_states, projections = cls._parse_projwfcout_body(
                data, parameters, atomic_states
            )
            parameters.update({"k_unit": "2pi/alat"})

        return cls(
            parameters,
            filename,
            proj_source="projwfc.out" if parse_projections else None,
            atomic_states=atomic_states,
            k=k,
            eigenvals=eigenvals,
        )

    @classmethod
    def from_filproj(cls, filename: str | os.PathLike, parse_projections: bool = True):
        """
        Construct a Projwfc object from a filproj file. This is the file
        generated by projwfc.x and is called `filproj.projwfc_up` by default.

        Spin-polarized calculations will also have a `filproj.projwfc_down` file. This method is intended to parse only one of the two files. If you want to parse both, you should parse them separately and then add them together as
            `projwfc_total = projwfc_up + projwfc_down`

        See the docstring of the `Projwfc.__add__` method for more information.

        Args:
            filename (str): Path to the file
            parse_projections (bool): Whether to parse the projections.
                If False, only the header of the file is read.
        """
        parameters, structure, skip = cls._parse_filproj_header(filename)

        nstates = parameters["natomwfc"]
        nkpnt = parameters["nkstot"]
        nbnd = parameters["nbnd"]
        noncolin = parameters["noncolin"]
        atomic_states = {}

        if parse_projections:
            # The length of an atomic state block in the filproj file
            nlines = nbnd * nkpnt + 1

            columns = np.arange(8) if noncolin else np.arange(7)
            data = pd.read_csv(
                filename,
                skiprows=skip,
                header=None,
                delim_whitespace=True,
                names=columns,
                dtype=str,
            )

            orbital_headers = data.values[::nlines, :]
            projections = data.values[:, 2]
            projections = np.delete(projections, slice(None, None, nlines))
            # k-point indices always run from 1 to nkpnt EXCEPT for the spin down
            # channel in spin polarized calculations (in filproj.projwfc_down)
            parameters["lsda"] = int(data.values[1, 0]) == nkpnt + 1
            spin = Spin.down if parameters["lsda"] else Spin.up
            projections = projections.reshape((nstates, nkpnt, nbnd), order="C").astype(
                float
            )

            # Process headers and save overlap data
            atomic_states = [None] * nstates
            for n in range(nstates):
                state_parameters = cls._parse_filproj_state_header(
                    orbital_headers[n], parameters, structure
                )
                atomic_states[n] = AtomicState(
                    state_parameters, {spin: projections[n, :, :]}
                )

        return cls(
            parameters,
            filename,
            proj_source="filproj" if parse_projections else None,
            atomic_states=atomic_states,
            structure=structure,
        )

    @classmethod
    def from_xml(
        cls,
        filename: str | os.PathLike,
        parse_eigenvals: bool = True,
        parse_k: bool = True,
        parse_projections: bool = True,
        selection: list[int] | bool = None,
        store_phi_psi: bool = False,
    ):
        """
        Constructs a Projwfc object from an atomic_proj.xml file. This uses a selective parsing method, where only the data requested is parsed. This is useful for large files where only a subset of the data is needed. However, please note that projwfc XML files are *not* symmetrized. Please see [this page](../../../../../../dev_notes/projwfc_output_comparison.md) for a comparison of the three files and some important details.

        Args:
            filename (str | os.PathLike): Path to the file
            parse_eigenvals (bool): Whether to parse the eigenvalues
            parse_k (bool): Whether to parse the k-points
            parse_projections (bool): Whether to parse the projections. Note that the
                XML does not actually contain the projections, but
                $\langle \phi | \psi \rangle$ where $\phi$ is the local orbital
                and $\psi$ is the Bloch function. The projections are computed as
                $|\langle \phi | \psi\rangle|^2$.
            selection (list): List of atomic states to parse. If None, all states
                are parsed. One indexed list (just like projwfc.x's ouptut)
            store_phi_psi (bool): Whether to store the raw data in the AtomicState
                objects. This will increase memory usage.
        """
        projections, phi_psi, eigenvals, k, weights, parameters = (
            cls._iterative_xml_parse(
                filename,
                parse_eigenvals,
                parse_k,
                parse_projections,
                selection,
                store_phi_psi,
            )
        )

        lsda = parameters["lsda"]
        natomwfc = parameters["natomwfc"]
        if selection is None:
            selection = np.arange(1, natomwfc + 1)

        # Create empty AtomicState objects for everything, only fill in parsed ones
        atomic_states = [AtomicState({"state_i": i + 1}) for i in np.arange(natomwfc)]
        if parse_projections:
            for state_i in selection - 1:
                state = atomic_states[state_i]
                state.projections[Spin.up] = projections[0, :, state_i, :]
                if lsda:
                    state.projections[Spin.down] = projections[1, :, state_i, :]
                if store_phi_psi:
                    state.phi_psi[Spin.up] = phi_psi[0, :, state_i, :]
                    if lsda:
                        state.phi_psi[Spin.down] = phi_psi[1, :, state_i, :]

        proj_source = "atomic_proj.xml" if parse_projections else None
        return cls(
            parameters,
            filename,
            proj_source,
            atomic_states=atomic_states,
            k=k,
            k_weights=weights,
            eigenvals=eigenvals,
        )

    @staticmethod
    def _iterative_xml_parse(
        filename, parse_k, parse_eigenvals, parse_projections, selection, store_phi_psi
    ):
        """
        Custom parser for atomic_proj.xml files. Allows for selective parsing of certain states.
        """
        header_parsed = False
        projections = []
        phi_psi = []
        k = []
        weights = []
        for event, elem in ET.iterparse(filename, events=("start", "end")):
            if not header_parsed and elem.tag == "HEADER" and event == "end":
                nbnd = int(elem.attrib["NUMBER_OF_BANDS"])
                nkstot = int(elem.attrib["NUMBER_OF_K-POINTS"])
                nspin = int(elem.attrib["NUMBER_OF_SPIN_COMPONENTS"])
                natomwfc = int(elem.attrib["NUMBER_OF_ATOMIC_WFC"])
                nelect = float(elem.attrib["NUMBER_OF_ELECTRONS"])
                efermi = float(elem.attrib["FERMI_ENERGY"]) * Ry_to_eV
                lsda = nspin == 2
                header_parsed = True
            elif header_parsed and elem.tag == "PROJECTIONS" and event == "end":
                if not selection:
                    selection = np.arange(1, natomwfc + 1)
                if parse_k:
                    k = np.zeros((nspin, nkstot, 3))
                    weights = np.zeros((nspin, nkstot))
                    for k_i, e in enumerate(elem.iter("K-POINT")):
                        k[k_i // nkstot, k_i % nkstot, :] = parse_pwvals(e.text)
                        weights[k_i // nkstot, k_i % nkstot] = float(e.attrib["Weight"])
                if parse_eigenvals:
                    eigen = np.zeros((nspin, nkstot, nbnd))
                    for k_i, e in enumerate(elem.iter("E")):
                        eigen[k_i // nkstot, k_i % nkstot, :] = parse_pwvals(e.text)
                if parse_projections:
                    k_i = 0
                    projections = np.zeros((nspin, nkstot, natomwfc, nbnd))
                    if store_phi_psi:
                        phi_psi = np.zeros(
                            (nspin, nkstot, natomwfc, nbnd), dtype=complex
                        )
                    k_i = 0
                    for e in elem.iter("ATOMIC_WFC"):
                        state_i = int(e.attrib["index"])
                        spin_i = int(e.attrib["spin"]) - 1
                        if state_i in selection:
                            p_p = np.array(parse_pwvals(e.text))
                            p_p = p_p[::2] + 1j * p_p[1::2]
                            projections[spin_i, k_i % nkstot, state_i - 1, :] = (
                                np.abs(p_p) ** 2
                            )
                            if store_phi_psi:
                                phi_psi[spin_i, k_i % nkstot, state_i - 1, :] = p_p
                        else:
                            projections[spin_i].append([])
                        if state_i == natomwfc:
                            k_i += 1

        if parse_k:
            if (
                lsda
                and not np.allclose(k[0, :], k[1, :])
                and not np.allclose(weights[0, :], weights[1, :])
            ):
                raise ProjwfcParserError(
                    "Spin up and down k-points and weights do not match. Something went wrong."
                )
            k = k[0, :]
            weights = weights[0, :]

        eigenvals = {}
        if parse_eigenvals:
            for spin_i in range(nspin):
                spin = Spin.up if spin_i == 0 else Spin.down
                eigenvals[spin] = eigen[spin_i] * Ry_to_eV

        parameters = {
            "natomwfc": natomwfc,
            "nbnd": nbnd,
            "nkstot": nkstot,
            "nelect": nelect,
            "efermi": efermi,
            "lsda": lsda,
            "k_unit": "2pi/alat",
        }

        return projections, phi_psi, eigenvals, k, weights, parameters

    @classmethod
    def _parse_projwfcout_header(cls, data):
        """
        Parse the header of a projwfc.out file. This is the stdout of projwfc.x.
        """
        state_header_regex = (
            r"\s*state #\s+(?P<state_i>\d+):\s+atom\s+(?P<atom_i>\d+)\s+"
            r"\((?P<species_symbol>\S+)\s*\)\s*,\s+wfc\s+(?P<wfc_i>\d+)\s+"
            r"\(l=\s*(?P<l>\d+)\s*(?:j=\s*(?P<j>\d+\.\d+)\s+m_j=\s*(?P<mj>[+-]?\d+\.\d+))?"
            r"\s*(?:m=\s*(?P<m>\d+))?\s*(?:s_z=\s*(?P<s_z>[+-]?\d+\.\d+))?\s*\)"
        )
        state_header_compile = re.compile(state_header_regex)

        natomwfc = int(re.findall(r"\s*natomwfc\s*=\s*(\d+)", data)[0])
        nx = int(re.findall(r"\s*nx\s*=\s*(\d+)", data)[0])
        nbnd = int(re.findall(r"\s*nbnd\s*=\s*(\d+)", data)[0])
        nkstot = int(re.findall(r"\s*nkstot\s*=\s*(\d+)", data)[0])
        npwx = int(re.findall(r"\s*npwx\s*=\s*(\d+)", data)[0])
        nkb = int(re.findall(r"\s*nkb\s*=\s*(\d+)", data)[0])

        atomic_states = []
        for state in state_header_compile.finditer(data):
            state_params = parse_pwvals(state.groupdict())
            site = Site(
                state_params["species_symbol"],
                [np.nan] * 3,
                properties={"atom_i": state_params["atom_i"], "Z": np.nan},
            )
            state_params.update({"site": site})
            for k, v in state_params.items():
                if v == "":
                    state_params[k] = None
            atomic_states.append(AtomicState(state_params))

        lspinorb = atomic_states[0].j is not None
        noncolin = atomic_states[0].s_z is not None or lspinorb
        # Both Noncolinear and spin-pol calcs include spin up and spin down channels
        if re.findall("\s*spin down", data) and not noncolin:
            # Spin pol calcs use twice the number of k-points, half for each spin
            nkstot //= 2
            lsda = True
        else:
            lsda = False

        parameters = {
            "natomwfc": natomwfc,
            "nx": nx,
            "nbnd": nbnd,
            "nkstot": nkstot,
            "npwx": npwx,
            "nkb": nkb,
            "lsda": lsda,
            "lspinorb": lspinorb,
            "noncolin": noncolin,
        }

        return parameters, atomic_states

    @classmethod
    def _parse_projwfcout_body(cls, data, parameters, atomic_states):
        """
        Parse the body of a projwfc.out file. This is the stdout of projwfc.x.
        """
        kpt_regex = (
            r"\s*k\s*=\s*(?P<kx>[+-]?\d+\.\d+)\s+(?P<ky>[+-]?\d+\.\d+)\s+"
            r"(?P<kz>[+-]?\d+\.\d+)\s*\n(?P<proj>\S.+?)(?:^$|\Z)"
        )
        state_regex = r"\s*(?P<proj>[+]?\d+.\d+)\*\[\#\s*(?P<state_i>\d+)\]\+?"
        band_regex = (
            r"\s*====\s*e\(\s*(?P<band_i>\d+)\)\s+=\s+(?P<eigenval>[+-]?\d+\.\d+)\s+eV\s===="
            r"\s*(?P<proj>.*?)\|psi\|\^2\s*=\s*(?P<psi2>\d+.\d+)"
        )

        band_compile = re.compile(band_regex, flags=re.MULTILINE | re.DOTALL)
        kpt_compile = re.compile(kpt_regex, re.MULTILINE | re.DOTALL)

        nkstot = parameters["nkstot"]
        natomwfc = parameters["natomwfc"]
        nbnd = parameters["nbnd"]

        nspin = 2 if parameters["lsda"] else 1
        k = np.zeros((nspin, nkstot, 3))
        eigenvals = {Spin.up: np.zeros((nkstot, nbnd))}
        if parameters["lsda"]:
            eigenvals[Spin.down] = np.zeros((nkstot, nbnd))
        projections = np.zeros((natomwfc, nspin, nkstot, nbnd))

        for i, kpt in enumerate(kpt_compile.finditer(data)):
            k_i = i % nkstot  # Accounts for LSDA
            spin_i = i // nkstot  # Accounts for LSDA
            k[spin_i, k_i] = parse_pwvals(list(kpt.groups()[:3]))
            spin = Spin.up if spin_i == 0 else Spin.down

            for band_i, band in enumerate(band_compile.finditer(kpt.groups()[3])):
                band_dict = band.groupdict()
                assert int(band_dict["band_i"]) == band_i + 1
                eigenvals[spin][k_i, band_i] = float(band_dict["eigenval"])
                proj_block = band_dict["proj"]
                for p in re.findall(state_regex, proj_block):
                    state_i = int(p[1])
                    proj = float(p[0])
                    projections[state_i - 1, spin_i, k_i, band_i] = proj

        if parameters["lsda"]:
            if not np.allclose(k[0, :], k[1, :]):
                raise ValueError(
                    "Spin up and spin down k-points are not the same in projwfc.out file."
                )
            k = k[0, :]

        for state_i, state in enumerate(atomic_states):
            state.projections[Spin.up] = projections[state_i, 0, :, :]
            if parameters["lsda"]:
                state.projections[Spin.down] = projections[state_i, 1, :, :]

        return k, eigenvals, atomic_states, projections

    @classmethod
    def _parse_filproj_state_header(cls, header, parameters, structure):
        """
        Parse the header of an atomic state in a filproj file.

        if noncolinear calculation with SOC:
           state_i atom_i species_symbol orbital_label wfc_i l j mj
        elif noncoliinear calculation without SOC:
           state_i atom_i species_symbol orbital_label wfc_i l m s_z
        if a colinear calculation (spin-polarized or not)
           state_i atom_i species_symbol orbital_label wfc_i l m

        See [this page](../../../../../../dev_notes/filproj_format.md) for more information on
        the format of the file.
        """

        header = parse_pwvals(header)
        atom_i = header[1]
        species_symbol = header[2]
        j = None
        mj = None
        m = None
        s_z = None
        if parameters["lspinorb"]:
            j = header[6]
            mj = header[7]
        elif parameters["noncolin"]:
            m = header[6]
            s_z = header[7]
        else:
            m = header[6]

        site = structure.sites[atom_i - 1]
        if site.species_string[:2] != species_symbol[:2]:
            raise ProjwfcParserError(
                "Species symbol in orbital header does not match species symbol in structure."
                " Something went wrong. "
            )

        return {
            "state_i": header[0],
            "wfc_i": header[4],
            "l": header[5],
            "j": j,
            "mj": mj,
            "m": m,
            "s_z": s_z,
            "n": int(header[3][0]),
            "site": site,
        }

    @classmethod
    def _parse_filproj_header(cls, filename):
        """
        Parse the header of a filproj file. See [this page](../../../../../../dev_notes/filproj_format.md)
        for more information on the format of the file.
        """
        with open(filename) as f:
            # First line is an empty line, skip it
            next(f)
            # Second line has format: nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp
            line = parse_pwvals(next(f))
            nrx = line[:3]
            nr = line[3:6]
            nat = line[6]
            ntyp = line[7]

            # Third line has format: ibrav celldm(1) ... celldm(6)
            line = parse_pwvals(next(f))
            ibrav = line[0]
            celldm = line[1:7]
            celldm[0] *= bohr_to_ang
            alat = celldm[0]

            # The next three lines are the lattice vectors, if ibrav = 0
            lattice = None
            if ibrav == 0:
                a1 = parse_pwvals(next(f))
                a2 = parse_pwvals(next(f))
                a3 = parse_pwvals(next(f))
                lattice_matrix = np.stack([a1, a2, a3]) * alat
                lattice = Lattice(lattice_matrix)
            else:
                lattice = ibrav_to_lattice(ibrav, celldm)
            # Next line is: gcutm dual ecutwfc 9 {last one is always 9}
            line = parse_pwvals(next(f))
            gcutm = line[0] * Ry_to_eV * (bohr_to_ang) ** 2
            dual = line[1]
            ecutwfc = line[2] * Ry_to_eV
            nine = line[3]

            # Next ntyp lines have format: species_i species_symbol nelect
            species_symbol = []
            nelect = []
            for _ in range(ntyp):
                line = parse_pwvals(next(f))
                species_symbol.append(line[1])
                nelect.append(line[2])

            # Next nat lines have format: atom_i x y z species_i
            species = [None] * nat
            Z = [None] * nat
            coords = np.zeros((nat, 3), float)
            for i in range(nat):
                line = parse_pwvals(next(f))
                if line[0] != i + 1:
                    raise ProjwfcParserError(
                        "Atom index (atom_i) in atomic coordinates section "
                        "of header does not match expected index. Something went wrong."
                    )
                coords[i] = np.array(line[1:4]) * alat
                species_i = line[4]
                species[i] = species_symbol[species_i - 1]
                Z[i] = nelect[species_i - 1]
            structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            structure.add_site_property("atom_i", range(1, nat + 1))
            # Add number of valence electrons as site property
            structure.add_site_property("Z", Z)

            # Next line has format: natomwfc nkstot nbnd
            line = parse_pwvals(next(f))
            natomwfc = line[0]
            nkstot = line[1]
            nbnd = line[2]

            # Next line has format: noncolin lspinorb (T or F)
            line = next(f).split()
            noncolin, lspinorb = map(lambda x: x == "T", line)

            header_nlines = ntyp + nat + 6
            if ibrav == 0:
                header_nlines += 3

            header = {
                "nrx": nrx,
                "nr": nr,
                "gcutm": gcutm,
                "dual": dual,
                "ecutwfc": ecutwfc,
                "nine": nine,
                "natomwfc": natomwfc,
                "nkstot": nkstot,
                "nbnd": nbnd,
                "noncolin": noncolin,
                "lspinorb": lspinorb,
            }

        return header, structure, header_nlines


class AtomicState(MSONable):
    """
    Class to store information about a single atomic state from projwfc.x or dox.x

    An atomic state is defined as an orbital one specific atom. The orbital is:
        - Defined by (n, l, m) if the calculation is colinear and not spin-polarized
        - Defined by (n, l, m) if the calculation is colinear spin-polarized. In this
            case, the spin up and spin down states are included in the same object.
        - Defined by (n, l, j, s_z) if the calculation is noncolinear without SOC.
        - Defined by (n, l, j, mj) if the calculation is noncolinear with SOC.

    Where:
    - n is the principal quantum number
    - l is the orbital or angular quantum number
    - m is the magnetic quantum number
    - j is the total angular momentum quantum number
    - m_j is the magnetic quantum number of the total angular momentum
    - s_z is the z component of the local spin (+- 1/2)

    QE also uses the notation of "wfc" to index the wavefunctions coming from a specific
    atom's pseudopotential. So these are unique l's or unique (l, j) pairs.

    Attributes:
        state_i (int): Index of this state, as indexed by projwfc.x
        wfc_i (int): Index of the wavefunction in the calculation
        l (int): Orbital angular momentum quantum number
        j (float): Total angular momentum quantum number
        mj (float): Magnetic quantum number of the total angular momentum
        s_z (float): S_z projection on a local z-axis (NCL calcs without SOC)
        m (int): Magnetic quantum number
        n (int): Principal quantum number
        site (pymatgen.core.structure.PeriodicSite): Site object for the atom
        orbital (pymatgen.electronic_structure.core.Orbital): Orbital object
        projections (dict): Projections of the state onto the atomic orbitals
        phi_psi (dict): Overlap of the state with the wavefunction
        pdos (dict): Projected density of states of the state
        energies (np.ndarray): Energies for the PDOS
    """

    def __init__(
        self,
        parameters: dict[str, any],
        projections: dict[Spin, np.ndarray] = None,
        phi_psi: dict[Spin, np.ndarray] = None,
        pdos: dict[Spin, np.ndarray] = None,
        energies: np.ndarray = None,
    ):
        """
        Initialize an AtomicState object.

        Args:

            parameters (dict): Dictionary with the following keys
                state_i (int): Index of this state, as indexed by projwfc.x
                wfc_i (int): Index of the wavefunction in the pseudopotential
                l (int): Orbital angular momentum quantum number
                j (float): Total angular momentum quantum number
                mj (float): Magnetic quantum number of the total angular momentum
                s_z (float): S_z projection on a local z-axis (NCL calcs without SOC)
                m (int): Magnetic quantum number
                n (int): Principal quantum number
                site (pymatgen.core.structure.PeriodicSite): Site object for the atom
            projections (dict[Spin, np.ndarray]): Projections from every band and
                k-point onto this atomic orbital.
            phi_psi (dict[Spin, np.ndarray]): Overlaps
            pdos (dict[Spin, np.ndarray]): Projected density of states
            orbital: (pymatgen.electronic_structure.core.Orbital): Orbital object
                Undefined for calculations with SOC.
        """
        self.state_i = parameters.get("state_i")
        self.wfc_i = parameters.get("wfc_i")
        self.l = parameters.get("l")
        self.j = parameters.get("j")
        self.mj = parameters.get("mj")
        self.s_z = parameters.get("s_z")
        self.m = parameters.get("m")
        self.n = parameters.get("n")
        self.site = parameters.get("site")
        self.orbital = None
        if self.l is not None and self.m is not None:
            self.orbital = Orbital(projwfc_orbital_to_vasp(self.l, self.m))
        self.projections = {} if projections is None else projections
        self.phi_psi = {} if phi_psi is None else phi_psi
        self.pdos = pdos
        self.energies = energies

    # TODO: need a better option
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """
        Equality test. This tests that the two objects represent the same state,
        i.e., same quantum numbers, state index, etc.
        """
        if not isinstance(other, AtomicState):
            return False

        return all(
            [
                self.state_i == other.state_i,
                self.wfc_i == other.wfc_i,
                self.l == other.l,
                self.j == other.j,
                self.mj == other.mj,
                self.s_z == other.s_z,
                self.m == other.m,
                self.n == other.n,
                self.site == other.site,
                self.orbital == other.orbital,
            ]
        )

    def __str__(self):
        out = []
        if self.l is not None:  # All fully parsed states (i.e., non-xml) have l
            out.extend(self._to_projwfc_state_string())
            atom_rep = " ".join(
                repr(self.site).split()[1:]
            )  # Get rid of "PeriodicSite: "
            out.append(f"Atom: {atom_rep}")
        else:
            out.append(f"State index: {str(self.state_i)} (not fully parsed)")

        return "\n".join(out)

    def _to_projwfc_state_string(self):
        """
        Returns an array with:
            1.  string representation of the state in the format printed by projwfc.x
                to stdout (with slight formatting improvements).
            2. A representation of the orbital (e.g., 5dxy)
        """
        state_rep = (
            f"state # {self.state_i:5d}:  atom {self.site.atom_i:5d} "
            f"({self.site.species_string}), wfc {self.wfc_i:5d} (l={self.l} "
        )
        if self.j is not None:
            state_rep += f"j={self.j} mj={self.mj:+})"
        elif self.s_z is not None:
            state_rep += f"m={self.m} s_z={self.s_z:+})"
        else:
            state_rep += f"m={self.m})"
        n = self.n or ""
        if self.orbital:
            if self.s_z:
                orbital_rep = f"Orbital: {n}{self.orbital} (s_z={self.s_z:+})"

            else:
                orbital_rep = f"Orbital: {n}{self.orbital}"
        else:
            orbital_rep = (
                f"Orbital: {n}{OrbitalType(self.l).name} (j={self.j}, mj={self.mj:+})"
            )
        return [state_rep, orbital_rep]


class ProjwfcParserError(Exception):
    """
    Exception class for Projwfc parsing.
    """


class InconsistentProjwfcDataError(Exception):
    """
    Exception class for Projwfc addition.
    """
