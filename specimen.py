"""Functions for generating a crystal specimen suitable for STEM simulations."""
import math

import ase
import numpy as np
from math import hypot as magnitude

from numpy.typing import NDArray
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Tuple

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_center(atoms: ase.Atoms, center: str = "COU") -> NDArray:
    """Return the center of the ASE Atoms object in cartesian coordinates.

    Arguments:
        atoms -- an ASE atoms object
        center -- the type of center; "cou" for center of cell, "com" for center of
                  mass, or "cop" for center of atom positions

    Adapted from ASE private function _centering_as_array(self, center)
    https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.rotate.
    """
    if center.lower() == "cou":
        return atoms.get_cell().sum(axis=0) / 2
    elif center.lower() == "com":
        return atoms.get_center_of_mass()
    elif center.lower() == "cop":
        return atoms.get_positions().mean(axis=0)
    else:
        raise ValueError("Cannot interpret center")


def center_atoms(atoms: ase.Atoms, center: str = "COU") -> None:
    """Move the atoms to the specified center point.

    Arguments:
        atoms -- an ASE atoms object
        center -- the type of center; "cou" for center of cell, "com" for center of
                  mass, or "cop" for center of atom positions
    Adapted from the ASE center() function
    https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.center.
    """
    center_coords = get_center(atoms, center)
    atoms.positions = atoms.positions - center_coords


def make_crystal_slab(
        material_structure,
        zone_axis: Tuple[int, int, int],
        slab_dimensions: Tuple[int, int, int],
        use_conventional_unit_cell: bool = True,
        grow_slab_to_fit_unit_cell: bool = False,
        zone_axis_rotation_angle: float = 0,
) -> ase.Atoms:
    """Create a bulk specimen of atoms with the crystal zone axis parallel to the z-axis.

     Arguments:
         material_structure -- a pymatgen structure with periodic boundary conditions
         zone_axis -- the crystal axis to align with the cartesian z-axis
         slab_dimensions -- the x-, y-, and z-dimensions of the desired crystal specimen
         use_conventional_unit_cell -- if True, the conventional cell is used, if False
                                      the primitive cell is used.  It is recommended to
                                      use the conventional cell so that the 001 zone
                                      axis for FCC crystals, for example, will appear
                                      to look as expected.
        grow_slab_to_fit_unit_cell -- if True, one or more of the slab_dimensions will
                                      be increased to ensure that the specimen contains
                                      at least one unit cell of material.  The final
                                      specimen dimensions will be contained in the 'cell'
                                      attribute of the returned ase.Atoms object.
        zone_axis_rotation_angle -- a value, in degrees, that specifies how much to
                                    rotate the crystal counterclockwise about the zone
                                    axis.  This rotation is applied after rotating the
                                    crystal to the zone axis.
    IMPORTANT: The zone_axis is specified in terms of crystallographic directions rather
    than cartesian directions.  For example, the crystal vector 001 always points along
    the c-axis of the crystal, but is not necessarily parallel to z-axis in cartesian
    space.

    First, repeat the atoms such that they completely fill an imaginary
    sphere whose radius is equal to or larger than the body diagonal of the
    desired specimen of atoms.  Doing so allows the crystal to be rotated to any direction
    while still filling up the entire specimen dimensions with atoms.

    Next, rotate the atoms to align the desired zone axis with the cartesian
    z-axis, because the z-axis is typically assumed to be axis along which electrons
    travel in (scanning) transmission electron microscope image simulations.

    Finally, delete any atoms outside the desired specimen dimensions and return the
    specimen as an ASE Atoms object.

    The returned ASE Atoms object will no longer have periodic boundary conditions.
    """
    if not material_structure:
        raise ValueError("No material structure provided.")

    # Convert pymatgen structure to ASE Atoms object

    if use_conventional_unit_cell:
        # structure is an instance of pymatgen.core.structure.Structure
        sga = SpacegroupAnalyzer(material_structure)
        structure = sga.get_conventional_standard_structure()
    else:
        structure = material_structure

    atoms = AseAtomsAdaptor.get_atoms(structure)

    if not np.all(atoms.pbc):
        raise ValueError(
            "The argument material_structure must have periodic boundary conditions."
        )

    # convert immutable parameters to numpy arrays
    slab_dims = np.array(slab_dimensions)
    za = np.array(zone_axis)

    if not np.all(slab_dims > 0):
        raise ValueError("All specimen dimensions must be larger than zero.")

    # Rotate the cell such that it is lower triangular, i.e., such that the first
    # crystal vector points entirely in the x-direction, the second crystal vector
    # lies in the xy-plane, and the third crystal vector points in the z-subspace.
    # The relationship is standard_cell @ rotation_matrix = cell,
    # where cell is the original cell,  standard_cell is the new standard
    # cell, and rotation_matrix rotates the standard cell back to
    # original cell. (NOTE: @ in numpy is the matrix multiplication operator)

    standard_cell, rotation_matrix = atoms.cell.standard_form()
    atoms.cell = standard_cell
    atoms.set_positions(atoms.positions @ np.linalg.inv(rotation_matrix))

    # Check whether the desired specimen will contain a full unit cell, regardless of how
    # the unit cell is rotated.

    if grow_slab_to_fit_unit_cell:
        cell_diagonal_vector = sum(atoms.cell)
        cell_diagonal_length = magnitude(*cell_diagonal_vector)

        if slab_dims[0] < cell_diagonal_length:
            slab_dims[0] = math.ceil(cell_diagonal_length)

        if slab_dims[1] < cell_diagonal_length:
            slab_dims[1] = math.ceil(cell_diagonal_length)

        if slab_dims[2] < cell_diagonal_length:
            slab_dims[2] = math.ceil(cell_diagonal_length)

    # Repeat the atoms such that they fill an imaginary sphere whose diameter is equal
    # to or greater than the length of the body diagonal dimension of the desired specimen.

    # Unpack the components of the slab_dims vector and calculate the length
    # of the specimen body diagonal.  This represents the minimum diameter of the imaginary
    # sphere to fill with atoms.

    diameter = magnitude(*slab_dims)

    # Get the projections of lattice vectors a, b, and c onto the cartesian directions
    # x, y, and z (e.g., "ax" is the projection of lattice vector "a" onto cartesian
    # axis "x"). Because the cell was converted to standard form above, these
    # projections are simply the x-component of a, the y-component of b, and the
    # z-component of c.  (Note: The "lattice vectors" are the three vectors of the
    # ASE atoms cell, which are specified in cartesian space).

    lattice_vectors_cartesian = atoms.get_cell()
    ax = lattice_vectors_cartesian[0][0]
    by = lattice_vectors_cartesian[1][1]
    cz = lattice_vectors_cartesian[2][2]

    # Calculate the number of times to repeat the atoms in each direction such that
    # the resultant group of atoms fills the imaginary sphere.

    repeat_x = math.ceil(diameter / ax)
    repeat_y = math.ceil(diameter / by)
    repeat_z = math.ceil(diameter / cz)

    # Repeat the atoms to fill the sphere

    repeated_atoms = atoms.repeat((repeat_x, repeat_y, repeat_z))

    # Convert the zone_axis from crystal coordinates to cartesian coordinates

    zone_axis_cartesian = np.array(
        za[0] * lattice_vectors_cartesian[0]
        + za[1] * lattice_vectors_cartesian[1]
        + za[2] * lattice_vectors_cartesian[2]
    )

    # Rotate the atoms to the zone axis.  No need to rotate the cell or specify center,
    # because we will recenter the atoms and create a new cell later.

    repeated_atoms.rotate(a=zone_axis_cartesian.tolist(),
                          v=[0, 0, 1],
                          center=(0, 0, 0),
                          rotate_cell=True)

    if zone_axis_rotation_angle:
        repeated_atoms.rotate(zone_axis_rotation_angle, v=[0, 0, 1])

    # Center the atoms in the cell (in general, the cell will be misaligned with the
    # atoms after rotation).

    repeated_atoms.center()

    # Center the atoms at the cartesian space origin (0, 0, 0)

    center_atoms(repeated_atoms, "cou")

    # We no longer have periodic boundary conditions because we sliced the atoms into a
    # specimen and arbitrarily set the cell to the specimen boundaries

    repeated_atoms.set_pbc((False, False, False))

    # Delete all atoms outside the specimen dimensions.  Slab of atoms must be centered
    # at 0, 0, 0 before this step.

    half_slab = slab_dims / 2

    del repeated_atoms[
        [
            atom.index
            for atom in repeated_atoms
            if (
                abs(atom.position[0]) > half_slab[0]
                or abs(atom.position[1]) > half_slab[1]
                or abs(atom.position[2]) > half_slab[2]
        )
        ]
    ]

    # Create a new cell that represents the specimen boundaries, then center the atoms in it
    repeated_atoms.set_cell(slab_dims)
    repeated_atoms.center()

    return repeated_atoms
