"""Tests for radius_graph_pbc_efficient self-loop handling.

A "home self-loop" is an edge connecting an atom to *itself* in the *home*
periodic image, i.e. ``sender == receiver`` AND ``cell_offset == [0, 0, 0]``.
Such an edge has a true length of exactly zero and must be removed, because a
zero-length edge produces ``0 / 0`` NaNs in the M3GNet three-body terms
(``cos_jik`` and the spherical-basis ``sin(c*r) / r``).

The self-loop filter must NOT remove *self-image* edges: an atom interacting
with its own periodic images (``sender == receiver`` but ``cell_offset != 0``)
is real physics and occurs whenever the cutoff exceeds the smallest lattice
spacing (e.g. small/primitive cells, or large cutoffs).

The historical filter discriminated home self-loops by ``distance <= 0.01``.
That distance is computed with ``torch.cdist``, whose default
``use_mm_for_euclid_dist`` mode is numerically unstable in float32 for
coordinates far from the origin: it can report a spurious ~0.011 Angstrom
distance between two *identical* points. That value exceeds the 0.01 threshold,
so the home self-loop survives and poisons the model with NaNs. These tests
pin the correct, offset-based behaviour.
"""

import itertools

import torch

from mattersim.datasets.utils.radius_graph_pbc import radius_graph_pbc_efficient

_PBC_TRUE = torch.tensor([[True, True, True]])


def _build_graph(pos: torch.Tensor, cell: torch.Tensor, cutoff: float):
    """Run the radius graph for a single periodic system."""
    return radius_graph_pbc_efficient(
        pos=pos,
        pbc=_PBC_TRUE,
        cell=cell,
        natoms=torch.tensor([pos.shape[0]]),
        radius=cutoff,
        max_num_neighbors_threshold=0,
        max_cell_images_per_dim=2147483647,
    )


def _home_self_loop_mask(
    edge_index: torch.Tensor, cell_offsets: torch.Tensor
) -> torch.Tensor:
    """Exact mask of home self-loops: same atom AND zero cell offset.

    ``cell_offsets`` are integer-valued (from ``torch.arange`` and
    ``torch.floor``), so the equality test is exact.
    """
    return (edge_index[0] == edge_index[1]) & (cell_offsets == 0).all(dim=1)


def _offset_grid_structure(
    n_per_dim: int = 7,
    box_len: float = 21.0,
    origin_shift_frac: float = 0.37,
    skew: float = 3e-5,
    dtype: torch.dtype = torch.float32,
):
    """A cubic grid of atoms placed far from the origin in a large cell.

    Chosen so that the wrapped Cartesian coordinates are large enough
    (|r| > ~20 Angstrom) that ``torch.cdist``'s float32 matmul path reports a
    spurious non-zero self-distance (~0.011 Angstrom) for several atoms. The
    tiny ``skew`` breaks exact cubic symmetry, mimicking a relaxed cell.
    """
    spacing = box_len / n_per_dim
    coords = torch.tensor(
        list(itertools.product(range(n_per_dim), repeat=3)), dtype=dtype
    )
    pos = coords * spacing + origin_shift_frac * box_len
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * box_len
    cell[0, 0, 1] = skew
    return pos, cell


def test_home_self_loops_are_removed_float32():
    """Regression: a far-from-origin float32 structure must yield NO
    home self-loop, despite torch.cdist's spurious self-distance.

    Without the offset-based filter this currently leaks several
    ``sender == receiver, offset == 0`` edges (true length zero), which the
    downstream M3GNet three-body terms turn into NaNs.
    """
    pos, cell = _offset_grid_structure(dtype=torch.float32)
    edge_index, cell_offsets, _, _, _ = _build_graph(pos, cell, cutoff=3.5)

    home_self_loops = _home_self_loop_mask(edge_index, cell_offsets)
    assert home_self_loops.sum().item() == 0, (
        f"{home_self_loops.sum().item()} home self-loop(s) leaked through the "
        "filter; these are zero-length edges that NaN the M3GNet three-body terms"
    )

    # Sanity: the graph is otherwise non-trivial (real neighbours present).
    assert edge_index.shape[1] > 0


def test_home_self_loops_absent_in_float64():
    """float64 has no cdist instability, so the float64 graph is the
    reference: it must also contain no home self-loops."""
    pos, cell = _offset_grid_structure(dtype=torch.float64)
    edge_index, cell_offsets, _, _, _ = _build_graph(pos, cell, cutoff=3.5)

    assert _home_self_loop_mask(edge_index, cell_offsets).sum().item() == 0


def test_self_image_edges_are_preserved():
    """An atom bonding to its own periodic images (sender == receiver but
    offset != 0) is real physics and must NOT be filtered out.

    A bcc Fe primitive cell (a = 2.87 Angstrom) is smaller than the 5 Angstrom
    cutoff, so every neighbour of the single atom is one of its own periodic
    images. Dropping these would leave the atom with zero neighbours.
    """
    a = 2.87
    cell = torch.tensor(
        [[[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]],
        dtype=torch.float64,
    )
    pos = torch.zeros((1, 3), dtype=torch.float64)

    edge_index, cell_offsets, _, _, distances = _build_graph(pos, cell, cutoff=5.0)

    # Every edge is a self-image edge (only one atom in the cell).
    same_atom = edge_index[0] == edge_index[1]
    assert same_atom.all(), "all bcc-primitive edges connect the atom to itself"

    # None of them is a *home* self-loop: all have a non-zero image offset...
    assert _home_self_loop_mask(edge_index, cell_offsets).sum().item() == 0
    # ...and all have a strictly positive interatomic distance.
    assert (distances > 0.1).all()

    # The physical neighbour shells must survive (bcc: 8 + 6 + ... within 5 A).
    assert edge_index.shape[1] >= 14
