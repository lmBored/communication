from escape_room.env_cfg import DEFAULT_NUM_ENVS
from escape_room.train import _is_cuda_oom, parse_args


def test_training_defaults_fit_a_40_gib_a100():
    args = parse_args([])

    assert args.task == "escape-room"
    assert args.num_envs == DEFAULT_NUM_ENVS == 8192
    assert not args.compile_game
    assert args.steps_per_update == 16
    assert args.physics_substeps == 1
    assert not args.check_nans


def test_warp_cuda_oom_is_recognized():
    error = RuntimeError("Graph launch error: Warp CUDA error 2: out of memory")

    assert _is_cuda_oom(error)
    assert not _is_cuda_oom(RuntimeError("unrelated failure"))