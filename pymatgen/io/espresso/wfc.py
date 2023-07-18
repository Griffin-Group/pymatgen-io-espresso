"""
Classes for reading and manipulating PWscf wave functions.
Currently only supports HDF5.
"""

from __future__ import annotations

import math
import os
import glob
import re

import numpy as np
import h5py

from pymatgen.core.units import (
    unitized,
    Ha_to_eV,
    Ry_to_eV,
    eV_to_Ha,
    bohr_to_ang,
    ang_to_bohr
)

from pymatgen.io.espresso.pwxml import PWxml
from pymatgen.io.espresso.utils import parse_pwvals 

class Wfc():
    # TODO: write docstring
    """
    Details...

    Attributes...

    Author: E. Banyas
    """

    def __init__(
        self,
        outdir,
        prefix,
        kids = None
    ):
        """
        Args:
            outdir: as specified in the PWscf calculation.
            prefix: as specified in the PWscf calculation.
            kids: an optional list of k-point indices to parse, e.g.
                `kids = [1,4,5]`.
                The default behavior is to parse wave functions at all
                k-points.
        """

        self.wfcdir = os.path.join(outdir,f'{prefix}.save')
        if not any(os.scandir(self.wfcdir)):
            # TODO: Best to throw an error now if the directory is empty?
            # (Conveniently, we also crash here if the directory does not exist)
            # TODO: add HDF5 vs. dat checking here 
            # temp:
            print("Oops, this directory is empty")
            sys.exit()

        fnames = glob.glob('wfc*.hdf5',root_dir=self.wfcdir)
        numbering = (
                lambda file: math.inf if not (match := re.match(
                    r".*?wfc(\d+).hdf5",file))
                else int(match.groups()[0])
                )
        files = sorted([os.path.join(self.wfcdir,fn) for fn in fnames], 
                       key = numbering)
        
        if kids is not None:
            # TODO: settle on strategy for ignoring k-pts ("None" approach OK?)
            try:
                kids = [ ki - 1 for ki in kids ]
                for idx, fn in enumerate(files):
                    if idx not in kids:
                        files[idx] = None
            except Exception:
                #TODO: proper error handling
                # For now, print a warning and parse all files
                print("Warning: could not parse the selected k-point indices")

        self.wfcs = [parseH5Wfc(f) for f in files]

class parseH5Wfc():
    def __init__(
        self,
        h5file
    ):
        if h5file == None:
            return None
        else:
            try:
                self.read_wfc(h5file)
            except Exception:
                print("Warning! Could not read the HDF5 file")
                return None
            #TODO: print a warning... and possibly set attributes to None?

    def read_wfc(self,h5file):
        #TODO: figure out exactly which units are used for evcs
        # (and WAVECARs, so that we can convert between the two)
        """
        Wave function format, as given in the QE developers' wiki:
        Attributes:
            'ik': k-point index
            'xk': k-point (cartesian) coordinates in Bohr^-1
            'nbnd': number of bands
            'ispin': spin index for LSDA (1 = spin-up, 2 = spin-down).
                For noncollinear OR unpolarized cases, ispin = 1.
            'npol': number of spin states (2 if noncollinear, 1 else)
            'igwx': number of plane-waves
            'ngw': number of plane-waves (defer to igwx)
            'gamma_only': if .true., only half of the plane waves are
                written
            'scale_factor': overall scale factor
        Data:
            'MillerIndices': (3,igwx) array of Miller indices
                Attributes:
                    'bg1', 'bg2', 'bg3': reciprocal lattice vectors in
                    Bohr^-1
            'evc': (2*npol*igwx,nbnd) array of wave function
                coefficients in atomic units.
                Note that the real and complex parts are
                stored in successive rows (hence the factor of 2).
                If npol = 2, the first igwx components correspond to
                the spin-up state, and the remainder are spin-down.
        """
        with h5py.File(h5file,'r') as f:
            #TODO: was the original encoding ascii or UTF-8?
            gam_str = f.attrs['gamma_only'].decode('ascii')
            self.is_gamma_only = parse_pwvals(gam_str)
            if f.attrs['npol'] == 2:
                self.n_pol = 2
                self.is_ncl = True
            else:
                self.n_pol = 1
                self.is_ncl = False
            self.k_index = f.attrs['ik']
            self.k_val = f.attrs['xk']
            self.n_bands = f.attrs['nbnd']
            self.spin_index = f.attrs['ispin']
            self.n_pw = f.attrs['igwx']
            self.scale = f.attrs['scale_factor']

            self.b_1 = f['MillerIndices'].attrs['bg1']
            self.b_2 = f['MillerIndices'].attrs['bg2']
            self.b_3 = f['MillerIndices'].attrs['bg3']

            # h5py reads this dataset with shape (igwx,3);
            # transpose to match Fortran convention
            self.millers = np.transpose(f['MillerIndices'][:,:])

            parsed_coefs = np.transpose(f['evc'][:,:])
            self.coefs = parsed_coefs[::2] + 1j*parsed_coefs[1::2]


#TODO:
# Test on gamma_only, LSDA, and noncollinear calculations
# (so far only tested on one vanilla calculation)

if __name__ == '__main__':
    test1 = Wfc('./tmp-tests/trial_dir/work','x',kids=[1,2])
    test2 = Wfc('./tmp-tests/trial_dir/work','x',kids="banana")
