import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import RewardConfig
from environments.blender_3d.rubric import Blender3DRubric


def _exec_result(*, f_score=0.09):
    return {
        "success": True,
        "mesh_stats": {
            "faces": 12,
            "vertices": 256,
        },
        "metrics": {
            "f_score_005": f_score,
        },
    }


def test_rubric_sub_rewards_are_binary():
    rubric = Blender3DRubric(RewardConfig())

    result = rubric.evaluate(
        (
            "import bpy\n"
            "bpy.ops.object.select_all(action='SELECT')\n"
            "bpy.ops.object.delete()\n"
            "bpy.ops.wm.obj_export(filepath='out.obj')\n"
        ),
        _exec_result(),
    )

    values = []
    for group in result["sub_rewards"].values():
        values.extend(group.values())

    assert values
    assert set(values).issubset({0.0, 1.0})
    assert result["sub_rewards"]["geometry"]["resemblance"] == 1.0
    assert result["sub_rewards"]["format"]["has_comments"] == 0.0


def test_rubric_thresholds_are_configurable():
    strict_cfg = RewardConfig()
    strict_cfg.format_reward_weight = 0.0
    strict_cfg.geometry.resemblance.threshold = 0.10

    relaxed_cfg = RewardConfig()
    relaxed_cfg.format_reward_weight = 0.0
    relaxed_cfg.geometry.resemblance.threshold = 0.08

    strict_result = Blender3DRubric(strict_cfg).evaluate(
        "import bpy\n# build object\n",
        _exec_result(f_score=0.09),
    )
    relaxed_result = Blender3DRubric(relaxed_cfg).evaluate(
        "import bpy\n# build object\n",
        _exec_result(f_score=0.09),
    )

    assert strict_result["sub_rewards"]["geometry"]["resemblance"] == 0.0
    assert relaxed_result["sub_rewards"]["geometry"]["resemblance"] == 1.0
    assert strict_result["base_reward"] == pytest.approx(6 / 7)
    assert relaxed_result["base_reward"] == pytest.approx(1.0)


def test_format_reward_affects_total_score():
    cfg = RewardConfig()
    cfg.geometric_weight = 0.0
    cfg.format_reward_weight = 1.0

    good = Blender3DRubric(cfg).evaluate(
        "import bpy\n# build object\nbpy.ops.object.select_all(action='SELECT')\n"
        "bpy.ops.object.delete()\nEXPORT_PATH='x'\n",
        _exec_result(),
    )
    poor = Blender3DRubric(cfg).evaluate("print('hi')\n", _exec_result())

    assert good["format_reward"] > poor["format_reward"]
    assert good["reward"] > poor["reward"]
