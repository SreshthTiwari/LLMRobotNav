import numpy as np

from final_proj.environment.context_extractor import ContextExtractor
from final_proj.environment.corruption import Corruptor


def test_map_loads_with_expected_cell_values(house_grid):
    assert house_grid.ndim == 2
    assert set(np.unique(house_grid)) <= {-1, 0, 1}
    assert (house_grid == 0).any() and (house_grid == 1).any()


def test_random_corruption_marks_requested_fraction(open_grid):
    np.random.seed(0)
    out = Corruptor(corruption_rate=0.2).inject_random_corruption(open_grid)
    assert (out == -1).sum() == int(open_grid.size * 0.2)
    assert (open_grid == 0).all(), "input grid must not be modified"


def test_corruption_sequence_has_one_grid_per_timestep(open_grid):
    np.random.seed(0)
    seq = Corruptor(0.1).inject_random_corruption(open_grid, change_over_time=True, num_timesteps=4)
    assert len(seq) == 4
    assert all(g.shape == open_grid.shape for g in seq)


def test_permanent_obstacle_is_identical_across_timesteps(open_grid):
    seq = Corruptor().generate_permanent_obstacle_sequence(open_grid, (2, 2, 5, 5), num_timesteps=3)
    assert len(seq) == 3
    for g in seq:
        np.testing.assert_array_equal(g, seq[0])
    assert (seq[0][2:5, 2:5] == 1).all()


def test_moving_obstacle_changes_between_timesteps(open_grid):
    path = [(x, 5) for x in range(10)]
    seq = Corruptor().generate_moving_obstacle_sequence(open_grid, path, num_timesteps=3)
    assert len(seq) == 3
    assert not np.array_equal(seq[0], seq[-1])


def test_patch_is_square_and_pads_out_of_bounds_as_blocked(open_grid):
    patch = ContextExtractor(patch_size=5).extract_patch(open_grid, (0, 0))
    assert len(patch) == 5 and all(len(r) == 5 for r in patch)
    assert patch[0][0] == 1  # off-map
    assert patch[2][2] == 0  # the centre cell


def test_find_uncertain_cells_returns_xy(open_grid):
    open_grid[3, 7] = -1
    assert ContextExtractor().find_uncertain_cells(open_grid) == [(7, 3)]
