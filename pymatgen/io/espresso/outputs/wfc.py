"""
Classes for reading and manipulating PWscf wave functions.
Currently only supports HDF5.
"""

#TODO:
# - .dat support
# "vaspify"
# classmethod, etc. decoration

from __future__ import annotations

import glob
import math
import os
import re
import sys

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
        if not os.path.isdir(self.wfcdir):
            raise OSError(
                    'Directory "outdir"/"prefix".save' \
                    f' ({self.wfcdir}) does not exist.'
                    )

        # TODO: is there a faster way to check this?
        self.exists_dat = False
        self.exists_hdf5 = False
        for file in os.listdir(self.wfcdir):
            if file.lower().endswith('dat'):
                self.exists_dat = True
            if file.lower().endswith('hdf5'):
                self.exists_hdf5 = True
                break

        if self.exists_hdf5:
            fnames = glob.glob('wfc*.hdf5',root_dir=self.wfcdir)
            numbering = (
                    lambda file: math.inf if not (match := re.match(
                        r".*?wfc(\d+).hdf5",file))
                    else int(match.groups()[0])
                    )
        elif self.exists_dat:
            # TODO: add support for .dat files
            print('ERROR: support for reading wfc.dat files is currently' \
                  ' not available. Supported filetypes are:\n' \
                  '  - wfc[k_n].hdf5')
            sys.exit()
        else:
            # TODO: raise exception
            print('ERROR: no wfc files found. Supported filetypes are:\n' \
                  '  - wfc[k_n].hdf5')
            sys.exit()

        files = sorted([os.path.join(self.wfcdir,fn) for fn in fnames], 
                       key = numbering)
        if kids is not None:
            try:
                kids = [ ki - 1 for ki in kids ]
                for idx, fn in enumerate(files):
                    if idx not in kids:
                        files[idx] = None
            except Exception:
                #TODO: proper error handling?
                print('Warning: could not parse the selected k-point' \
                        ' indices. All wfc files will be parsed instead.')

        if self.exists_hdf5:
            self.wfcs = [ParseH5Wfc(f) for f in files]
        else:
            # TODO: this is an empty class at the moment
            self.wfcs = [ParseDatWfc(f) for f in files]
            pass

class ParseH5Wfc():
    #TODO: docstring
    """
    Class for reading a provided wfc.hdf5 file.
    """
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
                print(f'Warning: could not read the HDF5 file at {h5file}.')
                return None
            #TODO: proper error handling?

    def read_wfc(self,h5file):
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
            # transpose to match Fortran convention:
            self.millers = np.transpose(f['MillerIndices'][:,:])

            parsed_coefs = np.transpose(f['evc'][:,:])
            self.coefs = parsed_coefs[::2] + 1j*parsed_coefs[1::2]

class ParseDatWfc():
    #TODO: everything
    """
    """
    def __init__(
        self,
        datfile
    ):
        if datfile == None:
            return None
        else:
            # TODO:
            return None

#TODO:
# Test on gamma_only, LSDA, and noncollinear calculations
# (so far only tested on one vanilla calculation)

if __name__ == '__main__':
    test1 = Wfc('./tmp-tests/trial_dir/work','x',kids='banana')
#    print('There are no files in here:')
#    test2 = Wfc('./tmp-tests/trial_dir/work','xyz')
#    print('There are only .dat files in here:')
#    test3 = Wfc('./tmp-tests/trial_dir/work','xyz-dat')
