import numpy as np

from final_proj.planning.astar import astar
from final_proj.planning.costmap import apply_region_penalty, apply_skill_bias, build_base_cost_map


def test_astar_finds_shortest_manhattan_path(open_grid):
    path = astar(build_base_cost_map(open_grid), (0, 0), (4, 3))
    assert path[0] == (0, 0) and path[-1] == (4, 3)
    assert len(path) == 4 + 3 + 1


def test_astar_steps_are_four_connected(open_grid):
    path = astar(build_base_cost_map(open_grid), (0, 0), (9, 9))
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert abs(x0 - x1) + abs(y0 - y1) == 1


def test_astar_returns_none_when_goal_walled_off(open_grid):
    open_grid[:, 5] = 1
    assert astar(build_base_cost_map(open_grid), (0, 0), (9, 9)) is None


def test_astar_routes_around_obstacle(open_grid):
    open_grid[:9, 5] = 1  # wall with a gap at the bottom
    path = astar(build_base_cost_map(open_grid), (0, 0), (9, 0))
    assert path is not None
    assert all(open_grid[y, x] == 0 for x, y in path)


def test_astar_prefers_free_over_uncertain_cells(open_grid):
    open_grid[0, 1:9] = -1  # uncertain corridor along the straight line
    path = astar(build_base_cost_map(open_grid), (0, 0), (9, 0))
    assert not any(open_grid[y, x] == -1 for x, y in path[1:-1])


def test_base_cost_map_values():
    grid = np.array([[0, 1, -1]])
    cost = build_base_cost_map(grid)
    assert cost[0, 0] == 1.0
    assert np.isinf(cost[0, 1])
    assert cost[0, 2] == 3.0


def test_skill_bias_lowers_cost_on_skill_path_only(open_grid):
    cost = build_base_cost_map(open_grid)
    biased = apply_skill_bias(cost, [(5, 5)], bias_strength=0.4)
    assert biased[5, 5] < cost[5, 5]
    assert biased[0, 0] == cost[0, 0]
    assert (biased <= cost).all()


def test_skill_bias_does_not_mutate_input(open_grid):
    cost = build_base_cost_map(open_grid)
    before = cost.copy()
    apply_skill_bias(cost, [(5, 5)])
    np.testing.assert_array_equal(cost, before)


def test_region_penalty_skips_blocked_and_out_of_bounds():
    cost = np.array([[1.0, np.inf]])
    out = apply_region_penalty(cost, [(0, 0), (1, 0), (5, 5)], penalty=5.0)
    assert out[0, 0] == 6.0
    assert np.isinf(out[0, 1])
