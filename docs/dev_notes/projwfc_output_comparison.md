# Comparison of `projwfc.x` output files

This document compares the output files of `projwfc.x` for the same calculation. The files are `projwfc.out` (just the `stdout` of `projwfc.x` redirected to a file), `filproj`, and `atomic_proj.xml`.

|                                                                                 | `projwfc.out`                         | `filproj` | `atomic_proj.xml`             |     |
| ------------------------------------------------------------------------------- | ------------------------------------- | --------- | ----------------------------- | --- |
| Structure                                                                       | ❌                                     | ✅         | ❌                             |     |
| $k$-points$^\text{[\small{Note 1}]}$                                            | ✅                                     | ❌         | ✅                             |     |
| Eigenvalues                                                                     | ✅                                     | ❌         | ✅                             |     |
| $\langle \psi_{nk} \vert \phi \rangle$$^{[\small{\text{Note 2}]}}$              | ❌                                     | ❌         | ✅                             |     |
| $\vert \langle\psi_{nk} \vert \phi\rangle \vert^2$                              | ✅$^{\text{\small{3 decimal places}}}$ | ✅         | ✅$^\text{\small{from above}}$ |     |
| Principle quantum number $n$ for $\vert\phi\rangle$$^{\small{[\text{Note 3}]}}$ | ❌                                     | ✅         | ❌                             |     |
| Orbital quantum numbers for $\vert\phi\rangle$$^{\small{[\text{Note 3}]}}$      | ✅                                     | ✅         | ❌                             |     |
| Partial Parsing$^{\small{[\text{Note 4}]}}$                                     | ❌                                     | ❌         | ✅                             |     |
| Same file for spin up/down$^{\small{[\text{Note 5}]}}$                          | ✅                                     | ❌         | ✅                             |     |
| Overlaps                                                                        | ✅                                     | ❌         | ❌                             |     |
| Lowdin Charges                                                                  | ✅                                     | ❌         | ❌                             |     |
| $E_{\text{Fermi}}$                                                              | ❌                                     | ❌         | ✅                             |     |
| Other Info$^{\small{[\text{Note 6}]}}$                                          | ❌                                     | ✅         | ❌                             |     |

**Note 1**: the $k$-points in `projwfc.out` and `atomic_proj.xml` are in cartesian coordinates in units of `alat`, and there's no way to convert them without reading the header of `filproj`
**Note 2:** It is necessary to know these if you want to obtain projections in the $(l, m_l, s_z)$ (or linear combinations thereof, e.g., $p_x$ spin up orbitals) for calculations with SOC.
**Note 3:** `projwfc.out` contains the quantum numbers ($(l, m)$ or $(l,m,s_z)$ or $(l, j, m_j)$) of the atomic states in the header, so it's easy to extract them. It doesn't contain $n$. They are spread throughout the file in `filproj` (which does contain $n$, so you have to parse the entire file to extract them. They are absent from `atomic_proj.xml`, only a numerical index for the atomic state is present and must be matched against one of the other files.
**Note 4:** You can extract only specific atomic states from `atomic_proj.xml` using an iterative XML parsing implementation. This can be a huge time save for large structures if you only care about certain orbitals ($O(10)$ minutes vs seconds to parse).
**Note 5**: For spin polarized calculations, you need `filproj.projwfc_up` and `filproj.projwfc_down`. However, `projwfc.out` in that case will include twice as many k-points, the first half of which is spin-up and the second half is spin-down, and it's not explicitly stated anywhere in the file that it's for a spin-polarized calculation (but you can tell because the Lowdin charges will mention spin up/down but the states won't have `s_z`)
**Note 6**: `filproj` contains other information such as the number of valence electrons (useful), cutoffs and FFT grids (might be useful for working with Lowdin charges?).

# Actual Data in the Files
Comparing the projection data in `filproj` and `projwfc.out`, they are quite similar but in general filproj is better since it has more precision. I noticed they more or less agree to within 5e-3 for non-spin polarized, LSDA (spin down only), noncolinear and SOC calculations. For the case of LSDA (spin up), there are some discrepancies but they are not that large. I don't understand why. 

Another note from my code:

```python
psi2 = float(band_dict["psi2"])
psi2_sum = np.sum(projections[:, k_i, band_i])
# The precision is so low in projwfc.out
# that they differ by 10-20%
if not np.isclose(psi2, psi2_sum, atol=1e-1):
     raise ValueError(
     "Sum of squared projections not "             "equal to |psi|^2 in projwfc.out "           f"file. {psi2} != {psi2_sum}"
     )
```


Note that the data in the XML [is not symmetrized](https://lists.quantum-espresso.org/pipermail/users/2022-April/048940.html). Further, see [this note](https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#idm54) about orthogonalization.
