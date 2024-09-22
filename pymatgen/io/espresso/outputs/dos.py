"""
Classes for reading/manipulating projwfc.x/dos.x (P)DOS files
"""

import itertools
import re

import numpy as np
import glob
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
        energies,
        tdensities,
        idensities=None,
        sum_pdensities=None,
        ldensities=None,
        atomic_states=None,
        efermi=None,
        lsda=None,
        noncolinear=None,
        lspinorb=None,
    ):
        """
        Args:
            energies (np.ndarray): energies in eV
            tdensities (Mapping[Spin, ArrayLike]): total DOS
            idensities (Mapping[Spin, ArrayLike]): integrated DOS
            sum_pdensities (Mapping[Spin, ArrayLike]): summed PDOS across all states
                                    Should be same as the total DOS but there's minor rounding
                                    Also this is spin polarized in noncolinear calcs but
                                    tdensities is not.
            ldensities (Mapping[Spin, ArrayLike]): local DOS for each wave function
                                             (Not lm or ljmj decomposed)
            atomic_states (list[AtomicState]): list of AtomicState objects
            efermi (float): Fermi energy
            lsda (bool | None): Whether the calculation is spin polarized.
                                None indicates unknown.
            noncolinear (bool | None): Whether the calculation is noncolinear
                                       (with or without SOC). None indicates unknown
            lspinorb (bool | None): Whether the calculation is spin orbit coupled
                                    None indicates unknown.
        """

        self.energies = energies
        self.tdensities = tdensities
        self.idensities = idensities
        self.ldensities = ldensities
        self.sum_pdensities = sum_pdensities
        self.atomic_states = atomic_states
        self.efermi = efermi
        self.noncolinear = noncolinear
        self.lsda = lsda
        self.lspinorb = lspinorb

    @classmethod
    def from_filpdos(cls, filpdos):
        """
        Args:
            filpdos (str): filepath to the pdos file. Note that this is
                the same as in projwfc.x input, so it shouldn't include the rest
                of the filename. For example, filpdos="path/to/filpdos" will look for
                "path/to/filpdos.pdos_atm#_wfc#..." type of files

            Whether the calculation is noncolinear WITHOUT SOC.
                                projwfc.x's output does not distinguish between
                                noncolinear without SOC and spin polarized calculations.
                                If False, the calculation is assumed to be spin polarized
                                (if the DOS file format leaves any ambiguity)
        """

        # Read the total DOS first
        # The only way to differentiate between spin polarized and noncolinear
        # without SOC is to check the number of columns in the fildos.pdos_tot file
        E, tdensities, sum_pdensities, ncl_no_soc, lsda = cls._read_total_pdos(
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
            raise ValueError("Energies from all files do not match.")
        energies = all_energies[0]
        # Order the atomic states and compute the state #
        atomic_states = cls._order_states(atomic_states)
        lspinorb = atomic_states[0].j is not None
        noncolinear = ncl_no_soc or lspinorb

        return cls(
            energies,
            tdensities,
            ldensities=ldensities,
            sum_pdensities=sum_pdensities,
            atomic_states=atomic_states,
            noncolinear=noncolinear,
            lsda=lsda,
            lspinorb=lspinorb,
        )

    @classmethod
    def from_fildos(cls, fildos):
        """
        Constructs a Dos object from a fildos (dos.x) file

        The format of Fildos is as follows (TDOS = total DOS, IDOS = integrated DOS):
        * Spin polarized:
            Energy(ev) TDOS(up) TODS(dn) IDOS
        * Everything else:
            Energy(ev) TDOS IDOS

        Args:
            filpdos (str): path to the dos file. Same as the dos.x input.
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
            raise ValueError(f"Unexpected number of columns {ncols} in {fildos}")

        return cls(
            energies, tdensities, idensities=idensities, efermi=efermi, lsda=lsda
        )

    @staticmethod
    def _order_states(atomic_states):
        """
        Sets the index (state #) of the states in the same order as the projwfc.x output
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
    def _read_total_pdos(filename):
        """
        Reads a filpdos.pdos_tot
        Returns the energies, total DOS, and summed PDOS



        * Colinear and noncolinear with SOC:
            Energy(eV) TDOS \sum_lm(PDOS_lm)
        * Colinear spin polarized:
            Energy(eV) TDOS(up) TDOS(dn) \sum_lm(PDOS_lm(up)) \sum_lm(PDOS_lm(dn))
        * Noncolinear without SOC:
            Energy(eV) TDOS \sum_lm(PDOS_lm(up)) \sum_lm(PDOS_lm(dn))

        Returns:
            energies (np.ndarray): energies in eV
            tdensities     (Mapping[Spin, ArrayLike]): total DOS
            sum_pdensities (Mapping[Spin, ArrayLike]): summed PDOS
            ncl_no_soc (bool): Whether the calculation is noncolinear *without* SOC
        """

        ncl_no_soc = False
        lsda = False

        data = np.loadtxt(filename, skiprows=1)
        energies = data[:, 0]
        if (ncols := data.shape[1]) == 3:
            # Colinear or noncolinear with SOC
            tdensities, sum_pdensities = {Spin.up: data[:, 1]}, {Spin.up: data[:, 2]}
        elif ncols == 4:
            # Noncolinear without SOC
            ncl_no_soc = True
            tdensities = {Spin.up: data[:, 1]}
            sum_pdensities = {Spin.up: data[:, 2], Spin.down: data[:, 3]}
        elif ncols == 5:
            # Colinear spin polarized
            lsda = True
            tdensities = {Spin.up: data[:, 1], Spin.down: data[:, 2]}
            sum_pdensities = {Spin.up: data[:, 3], Spin.down: data[:, 4]}
        else:
            raise ValueError(f"Unexpected number of columns {ncols} in {filename}")
        return energies, tdensities, sum_pdensities, ncl_no_soc, lsda

    @staticmethod
    def _read_pdos(filename, ncl_no_soc):
        """
        Gets an AtomicState object for each atomic state in the file

        The filename format is pdos_atm#<atom_i>(<symbol>)_wfc#<wfc_i>(<spdf>[_<j>])

        The number/order of columns is as follows:
            First column is always energy in eV, the rest depend on the calculation:
                Colinear: 1 for LDOS, 1 for each PDOS_m
                Colinear spin pol or noncolinear no SOC: 2 for LDOS (u/d), 2 for ea PDOS_m (u/d)
                Noncolinear with SOC: 1 for LDOS, 1 for each PDOS_mj
            LDOS = \sum_m PDOS_m or \sum_mj PDOS_mj
            The LDOS column should in principle be the same as the sum of the PDOS columns
            but finite precision may cause them to differ slightly

        params:
            filename (str): filename of the pdos file
            ncl_no_soc (bool): Whether the calculation is noncolinear without SOC
        returns:
            list of AtomicState objects
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
            ldos = {Spin.up: data[:, 1]}
            pdos = [{Spin.up: p} for p in data[:, 2:].T]
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "m": m}
                for m in np.arange(1, 2 * l + 2)
            ]
        elif ncols == (1 + 2 * (1 + 2 * l + 1)) and j is None:
            # Colinear spin polarized or noncolinear without SOC
            ldos = {Spin.up: data[:, 1], Spin.down: data[:, 2]}
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
            ldos = {Spin.up: data[:, 1]}
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "j": j, "mj": mj}
                for mj in np.arange(-j, j + 1)
            ]
        else:
            raise ValueError(
                f"Unexpected number of columns in {filename}. "
                "Are you trying to read k-resolved DOS? It's currently not implemented."
            )

        atomic_states = [AtomicState(pm, pdos=pd) for pm, pd in zip(params, pdos)]
        ldos = {"atom_i": site.atom_i, "l": OrbitalType(l), "j": j, "ldos": ldos}

        return energies, ldos, atomic_states
