"""
Classes for reading/manipulating projwfc.x/dos.x (P)DOS files
"""

import glob
import itertools
import os
import re

import numpy as np
from monty.json import MSONable

from pymatgen.core.structure import Site
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.io.espresso.outputs.projwfc import AtomicState


class EspressoDos(MSONable):
    """
    Class for representing DOS data from a dos.x or projwfc.x calculation
    """

    def __init__(
        self,
        energies: np.ndarray,
        tdos: dict[Spin, np.ndarray],
        idos: dict[Spin, np.ndarray] = None,
        summed_pdos: dict[Spin, np.ndarray] = None,
        summed_pdos_l: dict[Spin, list[np.ndarray]] = None,
        atomic_states: list[AtomicState] = None,
        efermi: float | None = None,
        lsda: bool | None = None,
        noncolinear: bool | None = None,
        lspinorb: bool | None = None,
    ):
        """
        Initializes an Espresso Dos object from a list of energies and densities
        of states. Shouldn't really be used directly unless you're doing something
        unconventional. Use the class method constructors from_filpdos (to parse
        projwfc.x outputs) or from_fildos (to parse dos.x output) instead.

        Args:
            energies (np.ndarray): Energies in eV. *Not* w.r.t the Fermi level.
            tdos (dict[Spin, np.ndarray]): Total DOS.
            idos (dict[Spin, np.ndarray], optional): Integrated DOS.
            summed_pdos (dict[Spin, np.ndarray], optional): Summed PDOS across
                all states (l,m) or (l,j, mj) etc. Should be the same as the total DOS
                but there might be minor rounding differences. This quantity is spin
                polarized in noncolinear calculations without SOC but tdos is not. This is essentially sum_{l,m} pdos_{l,m} or sum_{l,j,mj} pdos_{l,j,mj}.
            summed_pdos_l (dict[Spin, list[np.ndarray]], optional): Summed pDOS for each
                orbital, pretty much just \sum{m} pdos_{l,m} or \sum{mj} pdos_{l,j,mj}.
                Order is not guaranteed.
            atomic_states (list[AtomicState], optional): List of AtomicState objects.
                Order is guaranteed to be the same as projwfc.x output.
            efermi (float, optional): Fermi energy. # TODO: beter default
            lsda (bool, optional): Whether the calculation is spin polarized.
                None indicates unknown.
            noncolinear (bool, optional): Whether the calculation is noncolinear
                (with or without SOC). None indicates unknown.
            lspinorb (bool, optional): Whether the calculation includes spin-orbit
                coupling. None indicates unknown.

        Attributes:
            energies (np.ndarray): Energies in eV. *Not* w.r.t the Fermi level.
            tdos (dict[Spin, np.ndarray]): Total DOS.
            idos (dict[Spin, np.ndarray]): Integrated DOS.
            atomic_states (list[AtomicState]): Ordered list of AtomicState objects.
            efermi (float): Fermi energy.
            lsda (bool): Whether the calculation is spin polarized.
            noncolinear (bool): Whether the calculation is noncolinear.
            lspinorb (bool): Whether the calculation includes spin-orbit coupling.
        """

        self.energies = energies
        self.tdos = tdos
        self.idos = idos
        self.atomic_states = atomic_states  # Order guaranteed to be same as projwfc.x
        self.efermi = efermi
        self.noncolinear = noncolinear
        self.lsda = lsda
        self.lspinorb = lspinorb

        # Shouldn't be exposed to the user
        self._summed_pdos_l = summed_pdos_l  # Order is glob dependent
        self._summed_pdos = summed_pdos

    @classmethod
    def from_filpdos(cls, filpdos: str | os.pathlike) -> "EspressoDos":
        """
        Initialize an EspressoDos object from projwfc.x pdos files. This requires
        the filproj.pdos_tot and all filproj.pdos_atm#_wfc# files to be present.

        Args:
            filpdos (str | os.pathLike): path to the filproj pdos file. Note that this
                is the same quantity labeled "filproj" in projwfc.x's input, so it's not a full filename. For example, filpdos="path/to/filpdos" will look for files like "path/to/filpdos.pdos_atm#_wfc#...".
        """

        # Read the total DOS first. This is the only way to distinguish between
        # spin polarized calcs noncolinear calcs without SOC. We can't tell
        # colinear non-spin-polarized and noncolinear-with-SOC apart yet.
        E, tdos, summed_pdos, ncl_no_soc, lsda = cls._read_total_pdos(
            f"{filpdos}.pdos_tot"
        )
        all_energies = [E]

        atomic_states = []
        ldensities = []
        filenames = glob.glob(f"{filpdos}.pdos_atm*")
        for f in filenames:
            E, ldos, states = cls._read_pdos(f, ncl_no_soc)
            atomic_states.extend(states)
            ldensities.append(ldos)
            all_energies.append(E)
        if not np.allclose(all_energies, all_energies[0], rtol=0, atol=1e-4):
            raise WrongDosFormatError("Energies from all files do not match.")

        energies = all_energies[0]
        # Order the atomic states and compute the state #
        atomic_states = cls._order_states(atomic_states)
        lspinorb = atomic_states[0].j is not None
        noncolinear = ncl_no_soc or lspinorb

        return cls(
            energies,
            tdos,
            ldensities=ldensities,
            sum_pdensities=summed_pdos,
            atomic_states=atomic_states,
            noncolinear=noncolinear,
            lsda=lsda,
            lspinorb=lspinorb,
        )

    @classmethod
    def from_fildos(cls, fildos: str | os.pathlike) -> "EspressoDos":
        """
        Constructs a Dos object from a fildos (dos.x) file

        The format of Fildos is as follows (TDOS = total DOS, IDOS = integrated DOS):
        * Spin polarized:
            Energy(ev) TDOS(up) TODS(dn) IDOS
        * Everything else:
            Energy(ev) TDOS IDOS

        Args:
            fildos (str): path to the dos file. Same as the dos.x input.
        """
        with open(fildos, "r") as f:
            header = f.readline()
            if match := re.match(".*?EFermi\s*=\s*(\d+\.\d+)\s*eV.*?", header):
                efermi = float(match[1])
            else:
                raise ValueError("Cannot find Fermi energy in the header.")

        lsda = False

        data = np.loadtxt(fildos, skiprows=1)
        energies = data[:, 0]
        if (ncols := data.shape[1]) == 3:
            # Colinear or noncolinear (with or without SOC)
            tdensities, idensities = {Spin.up: data[:, 1]}, data[:, 2]
        elif ncols == 4:
            # spin polarized
            lsda = True
            tdensities = {Spin.up: data[:, 1], Spin.down: data[:, 2]}
            idensities = data[:, 3]
        else:
            raise WrongDosFormatError(
                f"Unexpected number of columns {ncols} in {fildos}"
            )

        return cls(
            energies, tdensities, idensities=idensities, efermi=efermi, lsda=lsda
        )

    @staticmethod
    def _order_states(atomic_states: list[AtomicState]) -> list[AtomicState]:
        """
        Sets the index (state #) of the states in the same order as the projwfc.x
        output. The sorting order is as follows:
            * atom_i, wfc_i, l, j, mj (spin orbit calculation)
            * atom_i, wfc_i, l, -s_z, m (noncolinear calculation, without SOC)
            * atom_i, wfc_i, l, m (colinear calculations, spin polarized or not)
        where atom_i is the index of the atom in the structure, wfc_i is the index of
        the "wavefunction"/orbital assigned by projwfc.x (read from the
        pseudopotential). All states with the same l, or (l, j) or are grouped together
        under one wfc index by QE.

        Args:
            atomic_states (list[AtomicState]): list of AtomicState objects
        Returns:
            list[AtomicState]: list of AtomicState objects, properly ordered
        """
        noncolinear = atomic_states[0].s_z is not None
        lspinorb = atomic_states[0].j is not None

        if lspinorb:

            def sort_order(x):
                return x.site.atom_i, x.wfc_i, x.l, x.j, x.mj
        elif noncolinear:

            def sort_order(x):
                return x.site.atom_i, x.wfc_i, x.l, -x.s_z, x.m
        else:

            def sort_order(x):
                return x.site.atom_i, x.wfc_i, x.l, x.m

        atomic_states = sorted(atomic_states, key=sort_order)
        for i, state in enumerate(atomic_states):
            state.state_i = i + 1

        return atomic_states

    @staticmethod
    def _read_total_pdos(
        filename: str | os.PathLike,
    ) -> tuple[np.ndarray, dict[Spin, np.ndarray], dict[Spin, np.ndarray], bool, bool]:
        """
        Reads a filpdos.pdos_tot file and returns the energies, total DOS, summed PDOS,
        and whether the calculation is noncolinear without SOC or spin-polarized.
        It is not possible to distinguish between colinear and SOC calculations from
        this file. The summed PDOS should, in principle, be the same as the total DOS,
        but minor rounding errors might occur.

        The format of the file is as follows:

        * Colinear and noncolinear with SOC:
            Energy(eV) TDOS \sum_lm(PDOS_lm)
        * Colinear spin polarized:
            Energy(eV) TDOS(up) TDOS(dn) \sum_lm(PDOS_lm(up)) \sum_lm(PDOS_lm(dn))
        * Noncolinear without SOC:
            Energy(eV) TDOS \sum_lm(PDOS_lm(up)) \sum_lm(PDOS_lm(dn))

        Args:
            filename (str | os.PathLike): Path to the pdos_tot file.

        Returns:
            tuple: A tuple containing:
            - energies (np.ndarray): Energies in eV, not w.r.t the Fermi level.
            - tdos (dict[Spin, np.ndarray]): Total DOS.
            - summed_pdos (dict[Spin, np.ndarray]): Summed PDOS.
            - ncl_no_soc (bool): Whether the calculation is noncolinear *without* SOC.
            - lsda (bool): Whether the calculation is spin polarized.

            If the last two booleans are False, the calculation is either colinear
            without spin polarization or noncolinear with SOC.
        """

        ncl_no_soc = False
        lsda = False

        data = np.loadtxt(filename, skiprows=1)
        energies = data[:, 0]
        if (ncols := data.shape[1]) == 3:
            # Colinear or noncolinear with SOC
            tdos, summed_pdos = {Spin.up: data[:, 1]}, {Spin.up: data[:, 2]}
        elif ncols == 4:
            # Noncolinear without SOC
            ncl_no_soc = True
            tdos = {Spin.up: data[:, 1]}
            summed_pdos = {Spin.up: data[:, 2], Spin.down: data[:, 3]}
        elif ncols == 5:
            # Colinear spin polarized
            lsda = True
            tdos = {Spin.up: data[:, 1], Spin.down: data[:, 2]}
            summed_pdos = {Spin.up: data[:, 3], Spin.down: data[:, 4]}
        else:
            raise ValueError(f"Unexpected number of columns {ncols} in {filename}")
        return energies, tdos, summed_pdos, ncl_no_soc, lsda

    @staticmethod
    def _read_pdos(
        filename: str | os.PathLike, ncl_no_soc: bool
    ) -> tuple[np.ndarray, dict[Spin, np.ndarray], list[AtomicState]]:
        """
        Parses a pdos files and returns the energies, summed PDOS, and a list of AtomicState objects for each state.

        The filename format is pdos_atm#<atom_i>(<symbol>)_wfc#<wfc_i>(<spdf>[_<j>])

        The number/order of columns is as follows:
            * Colinear calculations: For each orbital l, we have (2l + 1)+2 columns
                E (eV) LDOS PDOS_1 PDOS_2 ... PDOS_(2l+1)
            * Colinear spin polarized or noncolinear without SOC: For each orbital l,
                we have 2(2l + 2) + 1 columns
                E (eV) LDOS_up LDOS_dn PDOS_1up PDOS_1dn ... PDOS_(2l+1)up PDOS_(2l+1)dn
            * Noncolinear with SOC: For each orbital (l,j), we have (2j + 1)+2 columns
                E (eV) LDOS PDOS_1 PDOS_2 ... PDOS_(2j+1)
            Here, LDOS = \sum_m PDOS_m or \sum_mj PDOS_mj. The energy is in eV and
            *not* w.r.t the Fermi level, so that needs to be subtracted at some point.

            The LDOS column should in principle be the same as the sum of the individual
            PDOS columns, but finite precision may cause them to differ slightly.

        Args:
            filename (str | os.PathLike): Path to the pdos file.
            ncl_no_soc (bool): Whether the calculation is noncolinear without SOC.

        Returns:
            tuple: A tuple containing:
            - energies (np.ndarray): Energies in eV, not w.r.t the Fermi level.
            - ldos (dict[Spin, np.ndarray]): Summed PDOS.
            - atomic_states (list[AtomicState]): List of AtomicState objects.

        Raises:
            WrongDosFormatError: If the number of columns is unexpected
        """

        match = re.match(
            r".*?.pdos_atm#(\d+)\((\w+)\)_wfc#(\d+)\(([spdf])(?:_j(\d.5))*\)", filename
        )
        site = Site(match[2], [np.nan] * 3, properties={"atom_i": int(match[1])})
        wfc_i = int(match[3])
        l = OrbitalType[match[4]].value  # noqa: E741
        j = float(match[5]) if match[5] is not None else None

        data = np.loadtxt(filename, skiprows=1)
        energies = data[:, 0]
        if (ncols := data.shape[1]) == (2 + 2 * l + 1) and j is None:
            # Colinear case
            summed_pdos = {Spin.up: data[:, 1]}
            pdos = [{Spin.up: p} for p in data[:, 2:].T]
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "m": m}
                for m in np.arange(1, 2 * l + 2)
            ]
        elif ncols == (1 + 2 * (2 * l + 2)) and j is None:
            # Colinear spin polarized or noncolinear without SOC
            summed_pdos = {Spin.up: data[:, 1], Spin.down: data[:, 2]}
            if ncl_no_soc:
                pdos = [{Spin.up: p} for p in data[:, 3:].T]
                params = [
                    {"site": site, "wfc_i": wfc_i, "l": l, "m": m, "s_z": s_z}
                    for m, s_z in itertools.product(
                        np.arange(1, 2 * l + 2), [0.5, -0.5]
                    )
                ]
            else:
                pdos = [
                    {Spin.up: pu, Spin.down: pd}
                    for pu in data[:, 3::2].T
                    for pd in data[:, 4::2].T
                ]
                params = [
                    {"site": site, "wfc_i": wfc_i, "l": l, "m": m}
                    for m in np.arange(1, 2 * l + 2)
                ]
        elif j is not None and ncols == (2 + 2 * j + 1):
            # Noncolinear with SOC
            pdos = [{Spin.up: p} for p in data[:, 2:].T]
            summed_pdos = {Spin.up: data[:, 1]}
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "j": j, "mj": mj}
                for mj in np.arange(-j, j + 1)
            ]
        else:
            raise WrongDosFormatError(
                f"Unexpected number of columns in {filename}. "
                "Are you trying to read k-resolved DOS? It's currently not implemented."
            )

        atomic_states = [AtomicState(pm, pdos=pd) for pm, pd in zip(params, pdos)]
        summed_pdos = {
            "atom_i": site.atom_i,
            "l": OrbitalType(l),
            "j": j,
            "ldos": summed_pdos,
        }

        return energies, summed_pdos, atomic_states


class WrongDosFormatError(Exception):
    """
    Raised when the DOS file format is not recognized
    """

    pass
