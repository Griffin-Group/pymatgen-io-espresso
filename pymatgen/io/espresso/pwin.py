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
from io import StringIO
from copy import copy, deepcopy

import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
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

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.espresso.utils import parse_pwvals, ibrav_to_lattice


# TODO: implement conversion to VASP (and units for cutoffs, lattice constants, etc.)
class PWin(MSONable):
    """
    Class for PWscf input files
    """

    # First three are required, rest are optional
    _all_cards = [
        "atomic_species",
        "atomic_positions",
        "k_points",
        "additional_k_points",
        "cell_parameters",
        "constraints",
        "occupations",
        "atomic_velocities",
        "atomic_forces",
        "solvents",
        "hubbard",
    ]

    # Default options for each card
    # TODO: throw warning when default must be specified
    _all_defaults = [
        None,
        "alat",  # Not specifying option for atomic_positions is deprecated
        "tpiba",
        "tpiba",
        None,  # Option must be specified for cell_parameters
        None,  # constraints has no option
        None,  # occupations has no option
        "a.u.",  # a.u. is the only possible option
        None,  # atomic_forces has no option
        None,  # Option must be specified for solvents
        None,  # Option must be specified for hubbard
    ]
    _default_options = dict(zip(_all_cards, _all_defaults))

    # First three are required, rest are optional
    _all_namelists = [
        "control",
        "system",
        "electrons",
        "ions",
        "cell",
        "fcp",
        "rism",
    ]

    # TODO: doc string
    def __init__(self, namelists, cards, filename=None, bad_PWin_warning=True):
        """
        Args:
            namelists: dict of dicts of namelists
            cards: dict of dicts of cards
            filename (str): filename
            bad_PWin_warning (bool): Whether to warn if the PW input file is not
                valid (only a few checks currently implemented).
                Defaults to True.
        """
        self.bad_PWin_warning = bad_PWin_warning
        self.filename = filename

        self.control = namelists.get("control", None)
        self.system = namelists.get("system", None)
        self.electrons = namelists.get("electrons", None)
        self.ions = namelists.get("ions", None)
        self.cell = namelists.get("cell", None)
        self.fcp = namelists.get("fcp", None)
        self.rism = namelists.get("rism", None)

        self.atomic_species = cards.get("atomic_species", None)
        self.atomic_positions = cards.get("atomic_positions", None)
        self.k_points = cards.get("k_points", None)
        self.additional_k_points = cards.get("additional_k_points", None)
        self.cell_parameters = cards.get("cell_parameters", None)
        self.constraints = cards.get("constraints", None)
        self.occupations = cards.get("occupations", None)
        self.atomic_velocities = cards.get("atomic_velocities", None)
        self.atomic_forces = cards.get("atomic_forces", None)
        self.solvents = cards.get("solvents", None)
        self.hubbard = cards.get("hubbard", None)

        self._validate()

    @classmethod
    def from_file(cls, filename, suppress_bad_PWin_warn=False):
        """
        Reads a PWin from file

        Args:
            string (str): String to parse.
            suppress_bad_PWin_warn (bool): Whether to suppress warnings for bad PWin files.

        Returns:
            PWin object
        """
        parser = f90nml.Parser()
        parser.comment_tokens += "#"

        with open(filename) as f:
            pwi_str = f.read()

        namelists = parser.reads(pwi_str)
        namelists = namelists.todict()
        cards = cls._parse_cards(pwi_str)

        return cls(namelists, cards, filename, suppress_bad_PWin_warn)

    def to_str(self, indent=2):
        """
        Return the PWscf input file as a string
        """
        self._validate()
        namelists = {}
        # Some of the namelists can be {}, so we test against None instead of truthiness
        if self.control is not None:
            namelists.update({"control": self.control})
        if self.system is not None:
            namelists.update({"system": self.system})
        # Creating the namelist now helps preserve order for some reason
        namelists = f90nml.namelist.Namelist(namelists)  # type: ignore
        if self.electrons is not None:
            namelists.update({"electrons": self.electrons})
        if self.ions is not None:
            namelists.update({"ions": self.ions})
        if self.cell is not None:
            namelists.update({"cell": self.cell})
        if self.fcp is not None:
            namelists.update({"fcp": self.fcp})
        if self.rism is not None:
            namelists.update({"rism": self.rism})

        stream = StringIO()
        namelists.indent = indent
        namelists.write(stream)
        string = stream.getvalue()
        # Strip empty lines between namelists
        string = re.sub(r"\n\s*\n", "\n", string)

        string += self._card_to_str(self.atomic_species, indent)
        string += self._card_to_str(self.atomic_positions, indent)
        string += self._card_to_str(self.cell_parameters, indent)
        string += self._card_to_str(self.k_points, indent)
        string += self._card_to_str(self.additional_k_points, indent)
        string += self._card_to_str(self.constraints, indent)
        string += self._card_to_str(self.occupations, indent)
        string += self._card_to_str(self.atomic_velocities, indent)
        string += self._card_to_str(self.atomic_forces, indent)
        string += self._card_to_str(self.solvents, indent)
        string += self._card_to_str(self.hubbard, indent)

        return string

    def to_file(self, indent=2, filename="pw.in", overwrite=True):
        """
        Save the PWscf input file to a file
        """
        string = self.to_str(indent=indent)
        ascii = string.encode("ascii")
        if not overwrite and os.path.exists(filename):
            raise IOError("File exists! Use overwrite=True to force overwriting.")
        with open(filename, "wb") as f:
            f.write(ascii)

    def to_structure(self):
        """
        Returns:
            Structure object
        """
        # Step 1: get the lattice (in angstroms)

        # Step 2: get the atomic positions and species
        lattice = self._get_lattice()
        species, coords, coords_are_cartesian = self._get_atomic_positions()

        return Structure(lattice, species, coords, coords_are_cartesian=coords_are_cartesian)

    def _get_lattice(self):
        """
        Returns:
            Lattice object (in ANGSTROM no matter what's in the input file)
        """
        try:
            ibrav = self.system["ibrav"]
        except KeyError:
            raise ValueError("ibrav must be set in system namelist")
        if ibrav == 0:
            try:
                cell_parameters = self.cell_parameters
            except AttributeError:
                raise ValueError("cell_parameters must be set if ibrav=0")
            lattice_matrix = np.stack(
                (
                    cell_parameters["data"]["a1"],
                    cell_parameters["data"]["a2"],
                    cell_parameters["data"]["a3"],
                )
            )
            if cell_parameters["options"] == "alat":
                alat = self._get_alat("cell_parameters")
                lattice_matrix *= alat
            elif cell_parameters["options"] == "bohr":
                lattice_matrix *= bohr_to_ang
            elif cell_parameters["options"] == "angstrom":
                pass
            else:
                raise ValueError(
                    f"cell_parameters option must be one of 'alat', 'bohr', or 'angstrom'. {cell_parameters.option} is not supported."
                )
            lattice = Lattice(lattice_matrix)
        else:
            celldm = self._get_celldm()
            lattice = ibrav_to_lattice(ibrav, celldm)

        return lattice

    def _get_atomic_positions(self):
        """
        Parse the atomic positions from the atomic_positions card
        Returns:
            species (list): list of species symbols
            coords (ndarray): array of atomic coordinates (shape: (nat, 3)), ordered as in species
            coords_are_cartesian (bool): whether the coordinates are in cartesian or fractional coordinates. If true, then coords is in units of ANGSTROM, no matter what the units are in the input file.
        """
        try:
            atomic_positions = self.atomic_positions
        except AttributeError:
            raise ValueError("atomic_positions must be set")
        species = [x["symbol"] for x in atomic_positions["data"]]
        coords = np.array([x["position"] for x in atomic_positions["data"]])
        if atomic_positions["options"] == "alat":
            alat = self._get_alat("atomic_positions")
            coords *= alat
            coords_are_cartesian = True
        elif atomic_positions["options"] == "bohr":
            coords *= bohr_to_ang
            coords_are_cartesian = True
        elif atomic_positions["options"] == "angstrom":
            coords_are_cartesian = True
        elif atomic_positions["options"] == "crystal":
            coords_are_cartesian = False
        elif atomic_positions["options"] == "crystal_sg":
            raise ValueError("Atomic positions in crystal_sg option is not supported.")
        else:
            raise ValueError(
                f"atomic_positions option must be one of 'alat', 'bohr', 'angstrom', 'crystal', or 'crystal_sg'. {atomic_positions['options']} is not supported."
            )

        return species, coords, coords_are_cartesian

    def _get_alat(self, card_name):
        """
        Returns alat (either celldm(1) or A) in ANGSTROM with proper error handling
        """
        celldm = copy(self.system.get("celldm", None))
        A = self.system.get("A", None)
        if celldm is None and A is None:
            raise ValueError(f"either celldm(1) or A must be set if {card_name} option is alat")
        if celldm is not None and A is not None:
            raise ValueError("celldm(1) and A cannot both be set.")
        alat = celldm[0] * bohr_to_ang if celldm is not None else A

        return alat

    def _get_celldm(self):
        """
        Gets celldm from the input file.
        If celldm is in the input file, returns it with the first element converted to angstrom and padded with zeros to length 6.
        If A is in the input instead, then it returns:
            celldm = [A, B/A, C/A, cosBC, cosAC, cosAB] (with A in angstrom)
        with missing values padded to zeros

        Returns:
            celldm (list): list of celldm parameters, with shape (6,)
        """
        celldm = copy(self.system.get("celldm", None))
        A = self.system.get("A", None)
        if celldm is None and A is None:
            raise ValueError("either celldm(1) or A must be set if ibrav != 0")
        if celldm is not None and A is not None:
            raise ValueError("celldm(1) and A cannot both be set.")
        if celldm is not None:
            celldm[0] *= bohr_to_ang # celldm(1) is in bohr
            # Get it to the right length since not all are required in input
            celldm = np.pad(celldm, (0, 6 - len(celldm)))
        elif A is not None:
            # A is already in angstrom
            B = self.system.get("B", 0)
            C = self.system.get("C", 0)
            cosAB = self.system.get("cosAB", 0)
            cosAC = self.system.get("cosAC", 0)
            cosBC = self.system.get("cosBC", 0)
            celldm = [A, B / A, C / A, cosBC, cosAC, cosAB]

        return celldm

    @classmethod
    def _parse_cards(cls, pwi_str):
        cards_strs = pwi_str.rsplit("/", 1)[1].split("\n")
        cards_strs = [card for card in cards_strs if card]
        card_idx = [
            i for i, str in enumerate(cards_strs) if str.split()[0].lower() in cls._all_cards
        ]
        cards = {}
        for i, j in zip(card_idx, card_idx[1:] + [None]):  # type: ignore
            card_name = cards_strs[i]
            card_lines = cards_strs[i + 1 : j]
            cards[card_name] = card_lines
        found_cards = list(cards.keys())
        for c in found_cards:
            name = c.split()[0].lower()
            items = parse_pwvals(cards.pop(c))
            if len(c.split()) > 1:
                option = re.sub(r"[()]", "", c.split()[1])
                option = option.lower()
                option = re.sub(r"[()]", "", option)
                option = re.sub(r"[{}]", "", option)
            else:
                option = cls._default_options[name]
            cards[name] = cls._make_card(name, option, items)

        return cards

    @staticmethod
    def _make_card(name, options, data):
        card = {"name": name, "options": options}
        parsed_data = []
        if name == "atomic_species":
            for item in data:
                parsed_data.append({"symbol": item[0], "mass": item[1], "file": item[2]})
        elif name == "atomic_positions":
            for item in data:
                parsed_data.append({"symbol": item[0], "position": item[1:]})
        elif name == "cell_parameters":
            parsed_data = {"a1": data[0], "a2": data[1], "a3": data[2]}
        elif name == "k_points":
            if options == "automatic":
                k = data[0]
                parsed_data = {"grid": k[0:3], "shift": [bool(s) for s in k[3:]]}
            elif options == "gamma":
                parsed_data = None
            else:
                # Skip first item (number of k-points)
                for k in data[1:]:
                    if len(k) > 4:  # Then k = '0.0 0.0 0.0 1 ! label'
                        label = " ".join(k[4:]).strip("!").lstrip()
                    else:  # Then k = '0.0 0.0 0.0 1'
                        label = ""
                    parsed_data.append({"k": k[0:3], "weight": k[3], "label": label})
        else:
            # TODO: parse the other cards into a decent format
            parsed_data = data

        card["data"] = parsed_data
        return card

    @staticmethod
    def _card_to_str(card, indent):
        """
        Return the card as a string
        """
        if not card:
            return ""

        indent_str = " " * indent
        str = f"{card['name'].upper()}"
        if card["options"]:
            str += f" {{{card['options']}}}"
        if card["name"] == "atomic_species":
            for item in card["data"]:
                str += f"\n{indent_str}{item['symbol']:>3} {item['mass']:>10.6f} {item['file']}"
        elif card["name"] == "atomic_positions":
            for item in card["data"]:
                str += (
                    f"\n{indent_str}{item['symbol']:>3} {item['position'][0]:>13.10f}"
                    f" {item['position'][1]:>13.10f} {item['position'][2]:>13.10f}"
                )
        elif card["name"] == "cell_parameters":
            str += (
                f"\n{indent_str}{card['data']['a1'][0]:>13.10f}"
                f" {card['data']['a1'][1]:>13.10f}"
                f" {card['data']['a1'][2]:>13.10f}"
            )
            str += (
                f"\n{indent_str}{card['data']['a2'][0]:>13.10f}"
                f" {card['data']['a2'][1]:>13.10f}"
                f" {card['data']['a2'][2]:>13.10f}"
            )
            str += (
                f"\n{indent_str}{card['data']['a3'][0]:>13.10f}"
                f" {card['data']['a3'][1]:>13.10f}"
                f" {card['data']['a3'][2]:>13.10f}"
            )
        elif card["name"] == "k_points":
            if card["options"] == "automatic":
                str += (
                    f"\n{indent_str}{card['data']['grid'][0]:>3}"
                    f" {card['data']['grid'][1]:>3} {card['data']['grid'][2]:>3}"
                    f" {int(card['data']['shift'][0]):>3}"
                    f" {int(card['data']['shift'][1]):>3}"
                    f" {int(card['data']['shift'][2]):>3}"
                )
            elif card["options"] == "gamma":
                pass
            else:
                str += f"\n{len(card['data'])}"
                for item in card["data"]:
                    str += (
                        f"\n{indent_str}{item['k'][0]:>13.10f}"
                        f" {item['k'][1]:>13.10f} {item['k'][2]:>13.10f}"
                    )
                    # Check if weight is integer
                    if item["weight"] == int(item["weight"]):
                        str += f" {item['weight']:>4}"
                    else:
                        str += f" {item['weight']:>10.6f}"
                    if item["label"]:
                        str += f" ! {item['label']}"
        return str + "\n"

    def _validate(self):
        required_namelists = [self.control, self.system, self.electrons]
        if all(required_namelists):
            valid_namelists = True
        else:
            valid_namelists = False
            msg = "PWscf input file is missing required namelists:"
            for i, nml in enumerate(required_namelists):
                if not nml:
                    msg += f" &{self._all_namelists[i].upper()}"
            msg += ". Partial data available."
            if self.bad_PWin_warning:
                warnings.warn(msg, UserWarning)

        required_cards = [self.atomic_species, self.atomic_positions, self.k_points]
        if all(required_cards):
            valid_cards = True
        else:
            valid_cards = False
            msg = "PWscf input file is missing required cards:"
            for i, nml in enumerate(required_cards):
                if not nml:
                    msg += f" {self._all_cards[i].upper()}"
            msg += ". Partial data available."
            if self.bad_PWin_warning:
                warnings.warn(msg, UserWarning)

        return valid_namelists and valid_cards


class PWinParserError(Exception):
    """
    Exception class for PWin parsing.
    """
