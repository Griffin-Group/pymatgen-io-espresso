Credit for example and files: Ella Banyas, UC Berkeley/Physics and LBL/Molecular Foundry

This is a pathological yet not uncommon example where `celldm(1)` or `A` (i.e., `alat`) is equal to the bohr to angstrom conversion factor (or any other value that is not the true lattice constant), and (optionally) `CELL_PARAMETERS` and/or `ATOMIC_POSITIONS` is given in units of alat (i.e., angstrom).

This is explicitly warned against by the QE devs in the documentation, celldm(1) (or A) should *always* be the true lattice constant (in bohr or angstrom respectively). However, `pymatgen.io.espresso` can deal with this situation fine, although the behavior isn't extensively tested and is strongly not recommended.

# Details:
It gets confusing quite quickly. First, we need to define two values of alat:
`alat_true`: the lattice constant computed internally by Quantum ESPRESSO
`alat_fake`: the value of alat given in the input file

The XML contains two values of `alat` (or more if a relaxation calculation): `alat_fake` is in the **OUTPUT** section of the XML, even though it's supplied in the input file. `alat_true` is in the **INPUT** section of the XML, even though it's never given in the input and is computed internally. I am not sure why it's like this, but ultimately every section is consistent with its own `alat`.

In a well-posed calculation, `alat_fake = alat_true` and the value is unique (but repeated) in the XML file.

The units are as follows:
### Input Section of the XML
Recall that in this section, `alat = alat_true` = internally computed lattice constant.
* Crystal structure and atomic positions: units of bohr (no alat involved)
* k-points: units of `2pi/alat * ratio^2` where `alat=alat_true` and ratio = `alat_true/alat_fake`. Don't ask me why.

The k-points are only explicitly listed in the input section of the XML if the calculation used an explicit list of kpoints (i.e., the option in the `K_POINTS` card is not `{automatic}`). These explicit k-points are never read by Quantum ESPRESSO anyway.

### Output Section of the XML
Recall that in this section, `alat = alat_fake` = celldm(1) or A from input file (converted to Bohr)

* Crystal structure and atomic positions: units of bohr
* Reciprocal lattice vectors `b1,b2,b3`: units of `2pi/alat = 2pi/alat_fake`
* k-points: units of `2pi/alat = 2pi/alat_fake`

The k-points are always listed explicitly in the output section of the XML whether the calculation used a k-grid (i.e., `K_POINTS {automatic}`) or not.

# TL;DR:
You can use your files that set `celldm(1)` or `A` improperly with `pymatgen.io.espresso` but it's not recommended and hasn't been heavily tested. Listen to the QE manual and the warnings `pw.x` gives and set these flags properly.
