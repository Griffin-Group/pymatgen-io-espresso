"""
Utility functions for parsing Quantum ESPRESSO input and output files
"""

import math
import os
import re
import warnings
from glob import glob

import numpy as np

from pymatgen.core.lattice import Lattice


def parse_pwvals(
    val: str | dict | list | np.ndarray | None,
) -> str | dict[any, any] | list[any] | np.ndarray | bool | float | int:
    """
    Helper method to recursively parse values in the PWscf xml files. Supports array/list, dict, bool, float and int.

    Returns original string (or list of substrings) if no match is found.
    """
    # regex to match floats but not integers, including scientific notation
    float_regex = r"[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?"
    # regex to match just integers (signed or unsigned)
    int_regex = r"^(\+|-)?\d+$"
    if isinstance(val, dict):
        val = {k: parse_pwvals(v) for k, v in val.items()}
    elif isinstance(val, list):
        val = [parse_pwvals(x) for x in val]
    elif isinstance(val, np.ndarray):
        val = [parse_pwvals(x) for x in val]
        # Don't return as array unless all elements are same type
        if all(isinstance(x, type(val[0])) for x in val):
            val = np.array(val)
    elif val is None:
        val = None
    elif not isinstance(val, str):
        return val
    elif " " in val:
        val = [parse_pwvals(x) for x in val.split()]
    elif val.lower() in ("true", ".true."):
        val = True
    elif val.lower() in ("false", ".false."):
        val = False
    elif re.fullmatch(float_regex, val):
        val = float(val)
    elif re.fullmatch(int_regex, val):
        val = int(val)
    return val


def ibrav_to_lattice(ibrav, celldm):
    """
    Convert ibrav and celldm to lattice parameters.
    Essentially a reimplementation of latgen.f90
    See that module and the PW.x input documentation for more details.

    Args:
        ibrav (int): The ibrav value (see pw.x input documentation).
        celldm (list): The celldm values (see pw.x input documentation).

    Returns:
        Lattice: The lattice corresponding to the ibrav and celldm values.
    """
    warnings.warn(
        "ibrav != 0 has not been thoroughly tested. Please be careful.",
        IbravUntestedWarning,
    )
    _validate_celldm(ibrav, celldm)
    a = celldm[0]
    if ibrav == 0:
        raise ValueError("ibrav = 0 requires explicit lattice vectors.")
    elif ibrav == 1:
        # cubic P (sc)
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, a]
    elif ibrav == 2:
        # cubic F (fcc)
        a1 = [-a / 2, 0, a / 2]
        a2 = [0, a / 2, a / 2]
        a3 = [-a / 2, a / 2, 0]
    elif ibrav == 3:
        # cubic I (bcc)
        a1 = [a / 2, a / 2, a / 2]
        a2 = [-a / 2, a / 2, a / 2]
        a3 = [-a / 2, -a / 2, a / 2]
    elif ibrav == -3:
        # cubic I (bcc), more symmetric axis:
        a1 = [-a / 2, a / 2, a / 2]
        a2 = [a / 2, -a / 2, a / 2]
        a3 = [a / 2, a / 2, -a / 2]
    elif ibrav == 4:
        # Hexagonal and Trigonal P
        c = celldm[2] * a
        a1 = [a, 0, 0]
        a2 = [-a / 2, a * np.sqrt(3) / 2, 0]
        a3 = [0, 0, c]
    elif ibrav == 5:
        # Trigonal R, 3-fold axis c
        # The crystallographic vectors form a three-fold star around
        # the z-axis, the primitive cell is a simple rhombohedron.
        cos_g = celldm[3]  # cos(gamma)
        tx = np.sqrt((1 - cos_g) / 2)
        ty = np.sqrt((1 - cos_g) / 6)
        tz = np.sqrt((1 + 2 * cos_g) / 3)
        a1 = [a * tx, -a * ty, a * tz]
        a2 = [0, 2 * a * ty, a * tz]
        a3 = [-a * tx, -a * ty, a * tz]
    elif ibrav == -5:
        # Trigonal R, 3-fold axis (111);
        # The crystallographic vectors form a three-fold star around (111)
        a_p = a / np.sqrt(3)  # a'
        cos_g = celldm[3]  # cos(gamma)
        tx = np.sqrt((1 - cos_g) / 2)
        ty = np.sqrt((1 - cos_g) / 6)
        tz = np.sqrt((1 + 2 * cos_g) / 3)
        u = tz - 2 * np.sqrt(2) * ty
        v = tz + np.sqrt(2) * ty
        a1 = [a_p * u, a_p * v, a_p * v]
        a2 = [a_p * v, a_p * u, a_p * v]
        a3 = [a_p * v, a_p * v, a_p * u]
    elif ibrav == 6:
        # Tetragonal P (st)
        c = celldm[2] * a
        a1 = [a, 0, 0]
        a2 = [0, a, 0]
        a3 = [0, 0, c]
    elif ibrav == 7:
        # Tetragonal I (bct)
        c = celldm[2] * a
        a1 = [a / 2, -a / 2, c]
        a2 = [a / 2, a / 2, c]
        a3 = [-a / 2, -a / 2, c]
    elif ibrav == 8:
        # Orthorhombic P
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a, 0, 0]
        a2 = [0, b, 0]
        a3 = [0, 0, c]
    elif ibrav == 9:
        # Orthorhombic base-centered(bco)
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a / 2, b / 2, 0]
        a2 = [-a / 2, b / 2, 0]
        a3 = [0, 0, c]
    elif ibrav == -9:
        # Same as 9, alternate description
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a / 2, -b / 2, 0]
        a2 = [a / 2, b / 2, 0]
        a3 = [0, 0, c]
    elif ibrav == 91:
        # Orthorhombic one-face base-centered A-type
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a, 0, 0]
        a2 = [0, b / 2, -c / 2]
        a3 = [0, b / 2, c / 2]
    elif ibrav == 10:
        # Orthorhombic face-centered
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a / 2, 0, c / 2]
        a2 = [a / 2, b / 2, 0]
        a3 = [0, b / 2, c / 2]
    elif ibrav == 11:
        # Orthorhombic body-centered
        b = celldm[1] * a
        c = celldm[2] * a
        a1 = [a / 2, b / 2, c / 2]
        a2 = [-a / 2, b / 2, c / 2]
        a3 = [-a / 2, -b / 2, c / 2]
    elif ibrav == 12:
        # Monoclinic P, unique axis c
        b = celldm[1] * a
        c = celldm[2] * a
        cos_g = celldm[3]  # cos(gamma)
        sin_g = math.sqrt(1 - cos_g**2)
        a1 = [a, 0, 0]
        a2 = [b * cos_g, b * sin_g, 0]
        a3 = [0, 0, c]
    elif ibrav == -12:
        # Monoclinic P, unique axis b
        b = celldm[1] * a
        c = celldm[2] * a
        cos_b = celldm[4]  # cos(beta)
        sin_b = math.sqrt(1 - cos_b**2)  # sin(beta)
        a1 = [a, 0, 0]
        a2 = [0, b, 0]
        a3 = [c * cos_b, 0, c * sin_b]
    elif ibrav == 13:
        # Monoclinic base-centered (unique axis c)
        b = celldm[1] * a
        c = celldm[2] * a
        cos_g = celldm[3]  # cos(gamma)
        sin_g = math.sqrt(1 - cos_g**2)  # sin(gamma)
        a1 = [a / 2, 0, -c / 2]
        a2 = [b * cos_g, b * sin_g, 0]
        a3 = [a / 2, 0, c / 2]
    elif ibrav == -13:
        warnings.warn(
            "ibrav=-13 has a different definition in QE < v.6.4.1.\n"
            "Please check the documentation. The new definition in QE >= v.6.4.1 is "
            "used by pymatgen.io.espresso.\nThey are related by a1_old = -a2_new, "
            "a2_old = a1_new, a3_old = a3_new."
        )
        b = celldm[1] * a
        c = celldm[2] * a
        cos_b = celldm[4]  # cos(beta)
        sin_b = math.sqrt(1 - cos_b**2)  # sin(beta)
        a1 = [a / 2, b / 2, 0]
        a2 = [-a / 2, b / 2, 0]
        a3 = [c * cos_b, 0, c * sin_b]
    elif ibrav == 14:
        # Triclinic
        b = celldm[1] * a
        c = celldm[2] * a
        cos_g = celldm[3]  # cos(gamma)
        sin_g = math.sqrt(1 - cos_g**2)  # sin(gamma)
        cos_b = celldm[4]  # cos(beta)
        cos_a = celldm[5]  # cos(alpha)
        vol = np.sqrt(1 + 2 * cos_a * cos_b * cos_g - cos_a**2 - cos_b**2 - cos_g**2)

        a1 = [a, 0, 0]
        a2 = [b * cos_g, b * sin_g, 0]
        a3 = [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, c * vol / sin_g]
    else:
        raise ValueError(f"Unknown ibrav: {ibrav}.")

    lattice_matrix = np.array([a1, a2, a3])
    return Lattice(lattice_matrix)


def _validate_celldm(ibrav, celldm):
    """
    Validate the celldm array.
    """
    if len(celldm) != 6:
        raise ValueError(f"celldm must have dimension 6. Got {len(celldm)}.")
    if celldm[0] <= 0:
        raise ValueError(f"celldm[0]=a must be positive. Got {celldm[0]}.")
    if ibrav in (8, 9, 91, 10, 11, 12, -12, 13, -13, 14) and celldm[1] <= 0:
        raise ValueError(
            f"Need celldm[1]=b/a > 0 for ibrav = {ibrav}. Got {celldm[1]}."
        )
    if ibrav in (5, -5) and (celldm[3] <= -0.5 or celldm[3] >= 1.0):
        raise ValueError(
            f"Need -0.5 < celldm[3]=cos(alpha) < 1.0 for ibrav = {ibrav}. Got {celldm[3]}."
        )
    if ibrav in (4, 6, 7, 8, 9, 91, 10, 11, 12, -12, 13, -13, 14) and celldm[2] <= 0:
        raise ValueError(
            f"Need celldm[2]=c/a > 0 for ibrav = {ibrav}. Got {celldm[2]}."
        )
    if ibrav in (12, 13, 14) and abs(celldm[3]) > 1:
        raise ValueError(f"Need -1 < celldm[3]=cos(gamma) < 1. Got {celldm[3]}.")
    if ibrav in (-12, -13, 14) and abs(celldm[3]) > 1:
        raise ValueError(f"Need -1 < celldm[4]=cos(beta) < 1. Got {celldm[3]}.")
    if ibrav == 14:
        if abs(celldm[5]) > 1:
            raise ValueError(f"Need -1 < celldm[5]=cos(alpha) < 1. Got {celldm[5]}.")
        volume2 = (
            1
            + 2 * celldm[4] * celldm[5] * celldm[3]
            - celldm[4] ** 2
            - celldm[5] ** 2
            - celldm[3] ** 2
        )
        if volume2 <= 0:
            raise ValueError(
                f"celldm does not define a valid unit cell (volume^2 = {volume2} <= 0)."
            )


def projwfc_orbital_to_vasp(l: int, m: int):  # noqa: E741
    """
    Given l quantum number and "m" orbital index in projwfc output,
    convert to the orbital index in VASP (PROCAR).

    | orbital | QE (m/l) | VASP |
    |---------|----------|------|
    | s       | 0/1      |  0   |
    | pz      | 1/1      |  2   |
    | px      | 1/2      |  3   |
    | py      | 1/3      |  1   |
    | dz2     | 2/1      |  6   |
    | dxz     | 2/2      |  7   |
    | dyz     | 2/3      |  5   |
    | dx2     | 2/4      |  8   |
    | dxy     | 2/5      |  4   |

    """
    if l < 0 or l > 2:
        raise ValueError(f"l must be 0, 1, or 2. Got {l}.")
    if m < 1 or m > 2 * l + 1:
        raise ValueError(f"m must be between 1 and 2*l+1. Got {m}.")
    l_map = [[0], [2, 3, 1], [6, 7, 5, 8, 4]]
    return l_map[l][m - 1]


class FileGuesser:
    """
    Guesses a filename that matches the XML for a file of a specified filetype. This is done by combining the prefix from the XML with a set of extensions, as well as searching for some common file names, and searching in multiple directories near the XML file.


    | Filetype | Extensions | Set filenames | Search Directories |
    | -------- | ---------- | ------------- | ------------------ |
    | pwin     | `$prefix.in`, `$prefix.pwi`  | `bands.in`, `bands.pwi` | `./`, `../` |
    | filproj  | `$prefix`, `$prefix.proj` | `filproj` | `./`, `dos/`, `pdos/`, `projwfc`, `../pdos/`, `../dos/`, `../projwfc/` |
    | fildos   | `$prefix.dos` `$prefix.pdos` | `fildos`, `filpdos` | `./`, `dos/`, `../pdos/`, `projwfc/`, `../dos/`, `../projwfc/` |

    Returns filename for pwin, and filproj/fildos/filpdos for the other types.
    This means that it will return, for example, `dos/$prefix.proj` instead of
    `dos/$prefix.proj.projwfc_up` for filproj, and `dos/$prefix.dos` instead of
    `dos/$prefix.dos.pdos_tot` for filpdos.

    # TODO: include `outdir` in the guessing game.
    """

    def __init__(self, filetype, xml_filename, prefix):
        """
        Initializes the FileGuesser object with the filetype, XML filename, and prefix.

        Arguments:
            filetype (str): The type of file to guess. Can be "pwin", "filproj", "fildos", or "filpdos".
            xml_filename (str): The filename of the XML file to guess from.
            prefix (str): The prefix of the XML file.
        """
        self.filetype = filetype
        self.xml_filename = xml_filename
        self.prefix = prefix

        # Validate the filetype and set extensions, extras, and folders
        self._validate_filetype()

    def _validate_filetype(self):
        """
        Validates the filetype and sets the corresponding extensions, extras, and folders.
        """
        if self.filetype == "pwin":
            self.extensions = [".in", ".pwi"]
            self.extras = ["bands.in", "bands.pwi"]
            self.folders = ["../"]
        elif self.filetype == "filproj":
            self.extensions = ["", ".proj"]
            self.extras = ["filproj"]
            self.folders = ["dos", "pdos", "projwfc", "../pdos", "../dos", "../projwfc"]
        elif self.filetype in ["fildos", "filpdos"]:
            self.extensions = ["", ".dos", ".pdos"]
            self.extras = ["fildos", "filpdos"]
            self.folders = ["dos", "pdos", "../pdos", "../dos", "../projwfc"]
        else:
            raise ValueError(f"Unknown filetype to guess: {self.filetype}")

    def _generate_guesses(self):
        """
        Generates possible filename guesses based on the filetype, XML filename, prefix, and directories.

        Returns:
            list: A list of possible filename guesses.
        """
        basename = os.path.splitext(self.xml_filename)[0]
        dirname = os.path.dirname(self.xml_filename)
        guesses = [f"{basename}{ext}" for ext in self.extensions]
        guesses.extend(
            [os.path.join(dirname, f"{self.prefix}{ext}") for ext in self.extensions]
        )
        guesses.extend([os.path.join(dirname, f) for f in self.extras])

        if self.folders:
            guesses.extend(
                [
                    os.path.join(dirname, f, os.path.basename(g))
                    for f in self.folders
                    for g in guesses
                ]
            )

        return guesses

    def guess(self):
        """
        Guesses the appropriate filename based on the filetype, XML filename, and prefix.

        Returns:
            str: The guessed filename that matches the specified filetype.

        Raises:
            FileNotFoundError: If no appropriate file is found.
        """
        guesses = self._generate_guesses()
        print(f"All guesses for filetype = {self.filetype}")
        print(guesses)

        # Filter guesses based on filetype-specific rules
        if self.filetype == "filpdos":
            guesses = [g for g in guesses if glob(f"{g}.pdos_*")]
        elif self.filetype == "filproj":
            guesses = [g for g in guesses if glob(f"{g}.projwfc_*")]
        else:
            guesses = [g for g in guesses if os.path.exists(g)]

        if not guesses:
            raise FileNotFoundError(
                f"All guesses for an appropriate {self.filetype} file don't exist."
            )

        if len(set(guesses)) > 1:
            warnings.warn(
                f"Multiple possible guesses for {self.filetype} found. Using the first one: {guesses[0]}"
            )

        return guesses[0]


class IbravUntestedWarning(UserWarning):
    """
    Warning for untested ibrav values in ibrav_to_lattice
    and other related functions.
    """

    pass
