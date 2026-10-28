import json

import pytest

from final_proj.llm.client import LLMClient
from final_proj.llm.response_parser import ResponseParser
from final_proj.memory.similarity import best_matching_skill, compare_patches, similarity_score
from final_proj.memory.skill_store import SkillStore


def _context(patch_value=0, direction="east", turns=("L", "R"), length=20):
    return {
        "start_patch": [[patch_value] * 3 for _ in range(3)],
        "goal_direction": direction,
        "turn_pattern": list(turns),
        "path_length": length,
    }


def test_skill_store_persists_across_instances(tmp_path):
    path = tmp_path / "skills.json"
    store = SkillStore(path)
    skill_id = store.add_skill(_context(), _context(), [(0, 0), (1, 0)], "success")
    reloaded = SkillStore(path)
    assert reloaded.find_skill_by_id(skill_id)["outcome"] == "success"
    assert len(reloaded.get_all_skills()) == 1


def test_skill_ids_are_sequential(tmp_path):
    store = SkillStore(tmp_path / "s.json")
    ids = [store.add_skill({}, {}, [], "success") for _ in range(3)]
    assert ids == [0, 1, 2]
    assert store.find_skill_by_id(99) is None


def test_identical_contexts_score_one():
    assert similarity_score(_context(), _context()) == pytest.approx(1.0)


def test_different_contexts_score_lower():
    assert similarity_score(_context(), _context(patch_value=1, direction="west", turns=("R", "L"), length=90)) < 0.5


def test_compare_patches_handles_empty_and_mismatched_shapes():
    assert compare_patches([], [[0]]) == 0.0
    assert compare_patches([[0, 0]], [[0]]) == 0.0


def test_best_matching_skill_picks_closest():
    skills = [
        {"id": 0, "start_context": _context(patch_value=1, direction="west")},
        {"id": 1, "start_context": _context()},
    ]
    best, score = best_matching_skill(_context(), skills)
    assert best["id"] == 1 and score == pytest.approx(1.0)


def test_best_matching_skill_with_no_skills():
    assert best_matching_skill(_context(), []) == (None, -1.0)


def test_parse_temporal_response_fills_defaults():
    out = ResponseParser().parse_temporal_obstacle_response(json.dumps({"label": "moving"}))
    assert out["label"] == "moving"
    assert out["movement_pattern"] == "unknown"
    assert out["confidence"] == 0.0


def test_parse_region_response_coerces_null_confidence():
    out = ResponseParser().parse_region_response('{"label": "likely_free", "confidence": null}')
    assert out["confidence"] == 0.0


def test_parser_rejects_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        ResponseParser().parse_region_response("not json")


def test_client_falls_back_to_stub_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert LLMClient().is_stub
