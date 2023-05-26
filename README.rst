Pymatgen-analysis-myaddon
=========================

This is a template for writing an add-on for pymatgen. From v2022.0.3, pymatgen, pymatgen.analysis, pymatgen.ext and
and pymatgen.io are now `namespace packages <http://packaging.python.org/guides/packaging-namespace-packages/>`_. What
this means is that developers can now write packages that add functionality to pymatgen, such as:

* A new type of analysis (pymatgen.analysis);
* A high-level API access to a new external resource (pymatgen.ext); or
* Support for input/output from another code, e.g., a new quantum chemistry software (pymatgen.io).

For a real-world example using this template, check out `Materials Virtual Lab's pymatgen-analysis-diffusion
<http://github.com/materialsvirtuallab/pymatgen-analysis-diffusion>`_.

Usage
=====

1. Download this template as a zip file.
2. (Optional, Highly recommended) If you plan on using version control, create a Github or any other type of
   repository for your code and copy the contents of the zip file into your folder.
3. Rename all the `myaddon` in the template files to a name of your choosing.
4. Write your code. While you have complete freedom to organize your own code, there are a few rules that must be
   followed in order for your code to work properly as a pymatgen namespace package.

    a. The directory structure should be as follows::

        pymatgen
        # No __init__.py here. This is CRITICAL!
           analysis (alternatively, this can be ext or io)
           # No __init__.py here. This is CRITICAL!
              myaddon
              - __init__.py
              - other module.py files
    b. In setup.py, please name your package with the convention `pymatgen-<namespace>-<addon_name>`. Also make any
       modifications to the rest of the file, especially the `find_namespace_packages` call.
    c. Your minimum pymatgen dependency must be 2022.0.3. This is set by the line
       `install_requires=["pymatgen>=2022.0.3"]` in the setup.py and the requirements.txt file.

5. It is highly recommended you use the structure of this template and the included Github Actions workflows
   (see `.github </.github/workflows>`_ folder). These should work out of the box and performs linting and testing of
   your code with every push. Note that the testing is only as useful as the tests you write. It is highly recommended
   that you write unittests for all functionality.
6. Release your package on PyPi, see `Python Packaging Guide
   <http://packaging.python.org/tutorials/packaging-projects/>`_.
7. (Optional) You may submit a pull request to add your package to the `pymatgen add-ons listing page
   <https://github.com/materialsproject/pymatgen/blob/master/docs_rst/addons.rst>`_.

Afterwards, others will be able to install your package using::

    pip install pymatgen-<namespace>-<addon_name>

and all functionality will be available under::

    from pymatgen.<namespace>.<addon_name> import *
