# The format of `filproj`
## Preliminaries
* The header is written by `PP/src/write_io_header.f90`
* projections by `PP/src/write_proj.f90`, in the `write_proj_file` subroutine.

If the calculation is colinear (no spin polarization) or noncolinear, the filename will be `filproj.projwfc_up`. If it's spin-polarized colinear, you get a second file `filproj.projwfc_down`.


## Header Format
Header (stuff in `{}` are just my comments):
```
{empty line}
nr1x nr2x nr3x nr1 nr2 nr3 nat ntyp
ibrav celldm(1) ... celldm(6)
a1x a1y a1z {only if ibrav=0}
a2x a2y a2z {only if ibrav=0}
a3x a3y a3z {only if ibrav=0}
gcutm dual ecutwfc 9 {last one is always 9}
species_i species_symbol nelect {line repeated ntyp times}
atom_i x y z species_i {line repeated nat times}
natomwfc nkstot nbnd
noncolin lspinorb {only possible combinations: F F, T F, T T}
```

| Parameter                 | Explanation                                                                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nr1x, nr2x, nr3x`        | Physical dimensions of charge density array                                                                                                                                                                         |
| `nr1, nr2, nr3`           | (Dense) FFT grid dimensions/true dimensions of charge density array                                                                                                                                                                         |
| `nat`                     | Total number of atoms (same as `pw.x` input)                                                                                                                               |
| `ntyp`                    | Total number of species (same as `pw.x` input)                                                                                                                             |
| `ibrav`                   | Bravais lattice (same as `pw.x` input)                                                                                                                                     |
| `celldm(1) ... celldm(6)` | The lattice constants and angles $(a, b/a, c/a, \cos\alpha, \cos\beta, \cos\gamma)$ in bohr units (same as `pw.x` input). Only `celldm(1)` is guaranteed to be nonzero, the rest may be 0 depending on `ibrav`. |
| `a1x, a1y, a1z`           | x, y, z components of the first lattice vector in units of alat. Only present if `ibrav=0`.                                                                                |
| `a2x, a2y, a2z`           | x, y, z components of the second lattice vector in units of alat. Only present if `ibrav=0`.                                                                               |
| `a3x, a3y, a3z`           | x, y, z components of the second lattice vector in units of alat. Only present if `ibrav=0`.                                                                               |
| `gcutm`                   | $E_\text{cut}^{\rho}/(2\pi/a)^2$, where $E_\text{cut}^\rho$ = `ecutrho` and $a$ = `alat`,in units of $\text{Ry}\times a_0^2$.                                                                                                                                                                         |
| `dual`                    | `ecutrho/ecutwfc`                                                              |
| `ecutwfc`                 | The wavefunction cutoff in Ry (same as `pw.x` input)                                                                                                                       |
| `9`                       | The number 9. It's always there for an unknown reason                                                                                                                      |
| `species_i`           | Unique index (1 through `ntyp`) given to each unique species in the calculation                                                                                            |
| `species_symbol`          | Symbol for the element given in the input file (`O`, `Fe`)                                                                                                                 |
| `nelect`                  | Number of valence electrons in the pseudopotential file for the species.                                                                                                   |
| `atom_i`              | Unique index (1 through `nat`) given to each atom in the calculation                                                                                                       |
| `x y z`                   | Atom's position in cartesian coordinations, units of `alat` (regardless of what's in `pw.x`'s input). |
| `species_i`           | The index giving the type of the atom, see above.                                                                                                                           |
| `natomwfc`                | Number of total atomic wave functions (i.e., orbitals) available. Computed from the pseudopotential files                                                                  |
| `nkstot`                  | Number of total $k$-points in the calculation.                                                                                                                             |
| `nbnd`                    | Number of bands (same as `pw.x` input).                                                                                                                                    |
| `noncolin`                | Whether the calculation is noncolinear (`T` or `F`, same as `pw.x` input)                                                                                                     |
| `lspinorb`                | Whether the calculation includes spin-orbit coupling (`T` or `F`, same as `pw.x` input)                                                                                        |

# Projections
After the header, we end up with `natomwfc` sets of overlaps that schematically looks like this:
```
if noncolin and lspinorb:
    state_i atom_i species_symbol orbital_label wfc_i l j mj
elif noncolin and not lspinorb:
    state_i atom_i species_symbol orbital_label wfc_i l m s_z
else:
    state_i atom_i species_symbol orbital_label wfc_i l m 
for k_i in range(k_init, k_final+1):
    for band_i in range(1, nbnd+1):
      k_i band_i overlap
```
(i.e., there is a total of `nkstot*nbnd+1` lines)

| Parameter        | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `noncolin`       | Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                                                                             |     |
| `lspinorb`       | Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                                                                             |     |
| `state_i`        | A unique index for an atomic state, from 1 to `natomwfc`. Also written to `projwfc.x` `stdout`.                                                                                                                                                                                                                                                                                                                                                                             |     |
| `atom_i`         | Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                                                                             |     |
| `species_symbol` | Symbol for the species (e.g., `Fe`, `Sr`, etc.). Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                            |     |
| `orbital_label`  | A label for the atomic orbital, e.g., `4S` or `5P` or `3D`.                                                                                                                                                                                                                                                                                                                                                                                                                 |     |
| `wfc_i`      | A unique index for an atomic wfc of a given atom. A combination (`l`,`j`)or just `l` is considered unique. Prepended to the DOS files.                                                                                                                                                                                                                                                                                                                                    |     |
| `l`|Orbital quantum number $l$ | 
| `j`|Orbital quantum number $j$ | 
| `m`|Index for the type of orbital, see table below. Not the magnetic quantum number $m_l$. | 
| `s_z`| Eigenvalue of $\hat{S}_z$, $s_z = \pm 1/2$ | 
| `nkstot`         | Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                                                              |     |
| `k_init`, `k_final`         | The k indices run from 1 to nkstot for colinear spin-unpolarized, noncolinear and SOC calculations. For spin-polarized calculations, the indices run from `1` to `nkstot` for spin up and `nkstot+1` to `2*nkstot` for spin down.                                                                                                                                                                                                                                                                                                                                                                                                                               |     |
| `nbnd`           | Given in the header, see above.                                                                                                                                                                                                                                                                                                                                                                                                                                     |     |
| `overlap`        | $\vert\langle \phi_{\alpha}(\boldsymbol{\tau_i})\vert \psi_{n \boldsymbol{k}}^\sigma \rangle\vert^2$ where $\psi_{n \boldsymbol{k}}^\sigma$ is the Bloch wave of band $n$=`band_i` with $k$=`kpts[k_i]` (possibly with spin $\sigma$), and $\phi_\alpha(\boldsymbol{\tau}_i)$ is an atomic-like wavefunction centered at $\tau_i$, the position of the atom with $i$=`atom_i`, and $\alpha$ is the appropriate set of quantum numbers depending on the type of calculation. |     |

The orbitals are given by (from the `projwfc.x` input description):
*  $l=1$:

  | `m` | Orbital | Notes                                    |
  | ---- | ------- | ---------------------------------------- |
  | 1    | $p_z$      | $m_l=0$                                  |
  | 2    | $p_x$      | real combination of $m_l$=+/-1 with cosine |
  | 3    | $p_y$      | real combination of $m_l$=+/-1 with sine   |

* $l=2$:

| `m` | Orbital | Notes                                    |
| ---- | ------- | ---------------------------------------- |
|  1 | $d_{z^2}$    | $m_l=0$|
|  2 | $d_{zx}$    | real combination of $m_l=\pm1$ with cosine|
|  3 | $d_{zy}$    | real combination of $m_l=\pm1$ with sine|
|  4 | $d_{x^2-y^2}$ | real combination of $m_l=\pm2$ with cosine|
|  5 | $d_{xy}$    | real combination of $m=\pm2$ with sine|
* $l=3$ (not implemented in `projwfc.x`)
