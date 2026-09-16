from escape_room.consts import DEFAULT_CUBE_LAYOUT, DEFAULT_GAME_BACKEND
from escape_room.env_cfg import DEFAULT_NUM_ENVS
from escape_room.train import _is_cuda_oom, main, parse_args


def test_training_defaults_fit_a_40_gib_a100():
    args = parse_args([])

    assert args.task == "escape-room"
    assert args.num_envs == DEFAULT_NUM_ENVS == 32768
    assert args.steps_per_update == 16
    assert args.physics_substeps == 1
    assert not args.check_nans
    assert args.cube_layout == DEFAULT_CUBE_LAYOUT == "fixed"
    assert args.game_backend == DEFAULT_GAME_BACKEND == "warp"
    assert args.message_dim == 2


def test_communication_task_parser_contract():
    args = parse_args(["--task", "communication", "--message-dim", "4"])

    assert args.task == "communication"
    assert args.message_dim == 4


def test_communication_training_short_run(tmp_path, capsys):
    main(
        [
            "--task",
            "communication",
            "--device",
            "cpu",
            "--num-envs",
            "4",
            "--num-updates",
            "1",
            "--steps-per-update",
            "4",
            "--num-epochs",
            "1",
            "--num-minibatches",
            "1",
            "--num-channels",
            "32",
            "--ckpt-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert "task=communication" in output
    assert "trainer_env_SPS=" in output
    assert (tmp_path / "model_final.pt").is_file()


def test_warp_cuda_oom_is_recognized():
    error = RuntimeError("Graph launch error: Warp CUDA error 2: out of memory")

    assert _is_cuda_oom(error)
    assert not _is_cuda_oom(RuntimeError("unrelated failure"))