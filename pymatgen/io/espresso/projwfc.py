"""
Classes for reading/manipulating Projwfc.x files.
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

import numpy as np
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
import pandas as pd
from tabulate import tabulate

from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure, Site
from pymatgen.core.units import (
    unitized,
    Ha_to_eV,
    Ry_to_eV,
    eV_to_Ha,
    bohr_to_ang,
    ang_to_bohr,
)

from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.io.espresso.utils import parse_pwvals, ibrav_to_lattice, projwfc_orbital_to_vasp


class Projwfc(MSONable):
    """
    Class to parse projwfc.x output.
    """

    def __init__(self, parameters, structure, atomic_states, k, eigenvals, proj_source, filenames):
        self.parameters = parameters
        self.atomic_states = atomic_states
        self.structure = structure
        self.lspinorb = parameters["lspinorb"]
        self.noncolin = parameters["noncolin"]
        self.nstates = parameters["natomwfc"]
        self.nk = parameters["nkstot"]
        self.nbands = parameters["nbnd"]
        self.k = k
        self.eigenvals = eigenvals
        self.proj_source = proj_source
        self.filenames = filenames

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = []
        if self.lspinorb:
            header = f"Spin-orbit "
        elif self.noncolin:
            header = f"Noncolinear "
        else:
            if self.parameters["spin_down"]:
                header = "Colinear (spin-down) "
            else:
                header = "Colinear (spin-up or spin-unpolarized) "
        header += f"calculation with {self.nk} k-points and {self.nbands} bands."
        out.append(header)
        out.append("\n------------ Structure ------------")
        out.extend(str(self.structure).split("\n")[0:5])
        out.append(f"Sites ({self.structure.num_sites})")
        # Almost identical to Structure.__str__
        data = []
        for site in self.structure.sites:
            row = [site.atom_i, site.species_string]
            row.extend([f"{j:0.6f}" for j in site.frac_coords])
            row.append(site.Z)
            data.append(row)
        out.append(
            tabulate(
                data,
                headers=["#", "SP", "a", "b", "c", "Z val."],
            )
        )
        out.append("\n---------- Atomic States ----------")

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

        return "\n".join(out)

    class ProjwfcAtomicState(MSONable):
        """
        Class to store information about a single atomic state from Projwfc
        """

        def __init__(self, parameters, projections=None):
            self.state_i = parameters["state_i"]
            self.wfc_i = parameters["wfc_i"]
            self.l = parameters["l"]
            self.j = parameters["j"]
            self.mj = parameters["mj"]
            self.s_z = parameters["s_z"]
            self.m = parameters["m"]
            self.n = parameters.get("n", None)
            self.site = parameters.get("site", None)
            self.orbital = None
            if self.m:
                self.orbital = Orbital(projwfc_orbital_to_vasp(self.l, self.m))
            self.projections = projections

        def __repr__(self):
            return str(self)

        def __str__(self):
            out = []
            out.extend(self._to_projwfc_state_string())
            atom_rep = " ".join(repr(self.site).split()[1:])  # Get rid of "PeriodicSite: "
            out.append(f"Atom: {atom_rep}")

            return "\n".join(out)

        def _to_projwfc_state_string(self):
            """
            Returns an array with:
                1.  string representation of the state in the format used by projwfc.x stdout
                    (with slight formatting improvements).
                2. A representation of the orbital (e.g., 5dxy)
            """
            state_rep = (
                f"state # {self.state_i:5d}:  atom {self.site.atom_i:5d} "
                f"({self.site.species_string}), wfc {self.wfc_i:5d} (l={self.l} "
            )
            if self.j:
                state_rep += f"j={self.j} mj={self.mj:+})"
            elif self.s_z:
                state_rep += f"m={self.m} s_z={self.s_z:+})"
            else:
                state_rep += f"m={self.m})"
            n = self.n if self.n else ""
            if self.orbital:
                if self.s_z:
                    orbital_rep = f"Orbital: {n}{self.orbital} (s_z={self.s_z:+})"

                else:
                    orbital_rep = f"Orbital: {n}{self.orbital}"
            else:
                orbital_rep = f"Orbital: {n}{OrbitalType(self.l).name} (j={self.j}, mj={self.mj:+})"
            return [state_rep, orbital_rep]

    @classmethod
    def from_projwfcout(cls, filename, parse_projections=True):
        """
        Initialize from a projwfc.out file (stdout of projwfc.x)
        """

        with open(filename, "r") as f:
            if parse_projections:
                data = f.read()
            else:
                # TODO: better implementation
                # Does it matter how many lines you read
                # if you don't parse the projections?
                # Need benchmarking
                nlines = 1000
                head = list(itertools.islice(f, nlines))
                data = "\n".join(head)

        parameters, atomic_states = cls._parse_projwfcout_header(data)
        k, eigenvals = None, None
        if parse_projections:
            k, eigenvals, atomic_states = cls._parse_projwfcout_body(
                data, parameters, atomic_states
            )

        structure = None
        return cls(parameters, structure, atomic_states, k, eigenvals, "projwfc.out", [filename])

    @classmethod
    def from_filproj(cls, filename):
        """
        Initialize from a filproj file.
        """
        parameters, structure, skip = cls._parse_filproj_header(filename)

        nstates = parameters["natomwfc"]
        nkpnt = parameters["nkstot"]
        nbnd = parameters["nbnd"]
        noncolin = parameters["noncolin"]

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
        # k-point indices always run from 1 to nkpnt, EXCEPT in spin-polarized calculations
        # where they run from nkpnt+1 to 2*nkpnt for the spin down channel.
        spin_down = int(data.values[1, 0]) == nkpnt + 1
        parameters["spin_down"] = spin_down
        projections = projections.reshape((nstates, nkpnt, nbnd), order="C")

        # Process headers and save overlap data
        atomic_states = [None] * nstates
        for n in range(nstates):
            header = cls._parse_filproj_state_header(orbital_headers[n], parameters, structure)
            atomic_states[n] = cls.ProjwfcAtomicState(header, projections[n, :, :])

        return cls(parameters, structure, atomic_states, None, None, "filproj", [filename])

    @classmethod
    def _parse_projwfcout_header(cls, data):
        state_header_regex = (
            r"\s*state #\s+(?P<state_i>\d+):\s+atom\s+(?P<atom_i>\d+)\s+"
            r"\((?P<species_symbol>\S+)\s*\)\s*,\s+wfc\s+(?P<wfc_i>\d+)\s+"
            r"\(l=\s*(?P<l>\d+)\s*(?:j=\s*(?P<j>\d+\.\d+)\s+m_j=\s*(?P<mj>[+-]?\d+\.\d+))?"
            r"\s*(?:m=\s*(?P<m>\d+))?\s*(?:s_z=\s*(?P<s_z>[+-]?\d+\.\d+))?\s*\)"
        )
        state_header_compile = re.compile(state_header_regex)

        natomwfc = int(re.findall("\s*natomwfc\s*=\s*(\d+)", data)[0])
        nx = int(re.findall("\s*nx\s*=\s*(\d+)", data)[0])
        nbnd = int(re.findall("\s*nbnd\s*=\s*(\d+)", data)[0])
        nkstot = int(re.findall("\s*nkstot\s*=\s*(\d+)", data)[0])
        npwx = int(re.findall("\s*npwx\s*=\s*(\d+)", data)[0])
        nkb = int(re.findall("\s*nkb\s*=\s*(\d+)", data)[0])

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
            atomic_states.append(cls.ProjwfcAtomicState(state_params))

        parameters = {
            "natomwfc": natomwfc,
            "nx": nx,
            "nbnd": nbnd,
            "nkstot": nkstot,
            "npwx": npwx,
            "nkb": nkb,
            "lspinorb": atomic_states[0].j is not None,
            "noncolin": atomic_states[0].s_z is not None,
        }

        return parameters, atomic_states

    @classmethod
    def _parse_projwfcout_body(cls, data, parameters, atomic_states):
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

        k = np.zeros((nkstot, 3))
        eigenvals = np.zeros((nkstot, natomwfc))
        projections = np.zeros((natomwfc, nkstot, nbnd))

        for k_i, kpt in enumerate(kpt_compile.finditer(data)):
            k[k_i] = parse_pwvals(list(kpt.groups()[0:3]))
            for band_i, band in enumerate(band_compile.finditer(kpt.groups()[3])):
                band_dict = band.groupdict()
                assert int(band_dict["band_i"]) == band_i + 1
                proj_block = band_dict["proj"]
                for p in re.findall(state_regex, proj_block):
                    state_i = int(p[1])
                    proj = float(p[0])
                    projections[state_i - 1, k_i, band_i] = proj
                # psi2 = float(band_dict["psi2"])
                # psi2_sum = np.sum(projections[:, k_i, band_i])
                # The precision is so low in projwfc.out that they differ by 10-20%
                # if not np.isclose(psi2, psi2_sum, atol=1e-1):
                #     raise ValueError(
                #         "Sum of squared projections not equal to |psi|^2 in projwfc.out file. "
                #         f"{psi2} != {psi2_sum}"
                #     )
        for state_i, state in enumerate(atomic_states):
            state.projections = projections[state_i]

        return k, eigenvals, atomic_states

    @classmethod
    def _parse_filproj_state_header(cls, header, parameters, structure):
        # The format looks like this
        # if noncolin and lspinorb:
        #    state_i atom_i species_symbol orbital_label wfc_i l j mj
        # elif noncolin and not lspinorb:
        #    state_i atom_i species_symbol orbital_label wfc_i l m s_z
        # else:
        #    state_i atom_i species_symbol orbital_label wfc_i l m

        noncolin = parameters["noncolin"]
        lspinorb = parameters["lspinorb"]

        header = parse_pwvals(header)
        state_i = header[0]
        atom_i = header[1]
        species_symbol = header[2]
        n = int(header[3][0])
        wfc_i = header[4]
        l = header[5]

        j = None
        mj = None
        m = None
        s_z = None
        if noncolin and lspinorb:
            j = header[6]
            mj = header[7]
        elif noncolin and not lspinorb:
            m = header[6]
            s_z = header[7]
        else:
            m = header[6]

        site = structure.sites[atom_i - 1]
        # Strip off after the atomic symbol in case of, e.g., Fe2 in PWin becomes Fe2+ in structure
        if site.species_string[:2] != species_symbol[:2]:
            raise ProjwfcParserError(
                "Species symbol in orbital header does not match species symbol in structure."
                " Something went wrong."
            )

        return {
            "state_i": state_i,
            "wfc_i": wfc_i,
            "l": l,
            "j": j,
            "mj": mj,
            "m": m,
            "s_z": s_z,
            "n": n,
            "site": site,
        }

    @classmethod
    def _parse_filproj_header(cls, filename):
        # First line is an empty line, skip it
        # Second line has format: nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp
        with open(filename) as f:
            # First line is an empty line, skip it
            next(f)
            # Second line has format: nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp
            line = parse_pwvals(next(f))
            nrx = line[0:3]
            nr = line[3:6]
            nat = line[6]
            ntyp = line[7]

            # Third line has format: ibrav celldm(1) ... celldm(6)
            line = parse_pwvals(next(f))
            ibrav = line[0]
            celldm = line[1:7]
            celldm[0] *= bohr_to_ang
            alat = celldm[0]

            # The next three lines are the lattice constants if ibrav = 0, not there otherwise
            lattice = None
            if ibrav == 0:
                a1 = parse_pwvals(next(f))
                a2 = parse_pwvals(next(f))
                a3 = parse_pwvals(next(f))
                lattice_matrix = np.stack([a1, a2, a3]) * alat
                lattice = Lattice(lattice_matrix)
            else:
                lattice = ibrav_to_lattice(ibrav, celldm)
            # We then continue with a line with format: gcutm dual ecutwfc 9 {last one is always 9}
            line = parse_pwvals(next(f))
            gcutm = line[0] * Ry_to_eV * (bohr_to_ang) ** 2
            dual = line[1]
            ecutwfc = line[2] * Ry_to_eV
            nine = line[3]

            # Next ntyp lines have format: species_i species_symbol nelect
            species_symbol = []
            nelect = []
            for i in range(ntyp):
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
                        "Atom index (atom_i) in atomic coordinates section of header does not"
                        " match expected index. Something went wrong."
                    )
                coords[i] = np.array(line[1:4]) * alat
                species_i = line[4]
                species[i] = species_symbol[species_i - 1]
                Z[i] = nelect[species_i - 1]
            structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            structure.add_site_property("atom_i", range(1, nat + 1))
            # Add number of valence electrons as site property for future normalization purposes
            structure.add_site_property("Z", Z)

            # Next line has format: natomwfc nkstot nbnd
            line = parse_pwvals(next(f))
            natomwfc = line[0]
            nkstot = line[1]
            nbnd = line[2]

            # Next line has format: noncolin lspinorb
            line = parse_pwvals(next(f))
            noncolin = line[0]
            lspinorb = line[1]

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


class ProjwfcParserError(Exception):
    """
    Exception class for Projwfc parsing.
    """