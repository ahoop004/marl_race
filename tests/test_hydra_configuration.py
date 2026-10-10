"""Configuration migration parity and Hydra composition contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from application.configuration import validate_experiment_scenario
from core.configuration import PROJECT_ROOT, compose_configuration, source_path
from core.scenario import ScenarioError, load_and_expand_scenario, load_yaml_config
from training.algorithms import resolve_training_params
from training.evaluation_config import resolve_evaluation_protocol


PRETRAIN = "scenarios/ppo_lap_completion_pretrain.yaml"
TEAM = "scenarios/mappo_2v2_completion_scratch.yaml"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parity_fingerprints(config, source, params=None):
    """Compare task values and effective learner inputs across schema changes."""
    from agents.common.networks import resolve_network_config

    def canonical_path(path):
        return "$PROJECT_ROOT/" + Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()

    task = {k: deepcopy(config[k]) for k in ("experiment", "environment", "agents", "evaluation", "wandb")
            if k in config}
    if "mappo" in config:
        task["mappo"] = deepcopy(config["mappo"])
    env = task["environment"]
    env["map_dir"] = canonical_path(source_path(env.get("map_dir", "maps")))
    if env.get("max_speed") is None:
        env.pop("max_speed", None)
    if task["experiment"].get("checkpoint"):
        task["experiment"]["checkpoint"] = canonical_path(source.parent / task["experiment"]["checkpoint"])
    observation_order = {}
    learners = {}
    for aid, agent in task["agents"].items():
        if not agent.get("trainable"):
            continue
        agent.pop("params", None)
        observation_order[aid] = list(agent["observation"]["observation"])
        values = deepcopy(params[aid] if params is not None
                          else resolve_training_params(config["agents"][aid], config))
        values["network"] = resolve_network_config(values, default_hidden_dims=[64, 64])
        for k in ("pi_hidden_dims", "vf_hidden_dims", "hidden_dims", "activation"):
            values.pop(k, None)
        if values.get("pretrained_actor_checkpoint"):
            values["pretrained_actor_checkpoint"] = canonical_path(source.parent / values["pretrained_actor_checkpoint"])
        learners[aid] = values
    return {"task": digest(task), "learner": digest(learners), "observation_order": observation_order}


GOLDEN = json.loads((Path(__file__).parent / "fixtures/configuration_parity.json").read_text())


@pytest.mark.parametrize("path", GOLDEN)
def test_all_existing_presets_preserve_effective_configuration(path):
    config = load_and_expand_scenario(path)
    validate_experiment_scenario(config)
    assert parity_fingerprints(config, source_path(path)) == GOLDEN[path]
    assert "includes" not in config
    assert config["_configuration"]["source"] == str(source_path(path))


def test_group_choices_select_algorithms_tasks_and_networks_independently():
    config = load_and_expand_scenario(PRETRAIN, overrides=[
        "algorithm=mappo", "scenario=cooperative_2v2_completion",
        "execution=mappo", "evaluation=team_completion", "maps=team_completion",
        "network.actor_hidden_dims=[32,16]", "network.critic_hidden_dims=[64]",
        "network.actor.parameter_sharing=shared", "network.activation=relu",
    ])
    validate_experiment_scenario(config)
    assert config["_configuration"]["choices"]["algorithm"] == "mappo"
    assert config["_configuration"]["choices"]["observation"] == "team"
    assert [a["algorithm"] for a in config["agents"].values()] == ["mappo", "mappo", "racing_mpc", "racing_mpc"]
    assert config["mappo"]["actor_mode"] == "shared"
    params = resolve_training_params(config["agents"]["car_0"], config)
    assert params["network"] == dict(architecture="mlp", actor_hidden_dims=[32,16],
                                      critic_hidden_dims=[64], activation="relu")
    assert params["n_steps"] == 2048 and params["batch_size"] == 512


@pytest.mark.parametrize("adaptation", ["scratch", "full_finetune", "lora"])
def test_adaptation_choices_do_not_change_the_task(adaptation):
    scratch = load_and_expand_scenario(TEAM)
    config = load_and_expand_scenario(TEAM, overrides=[f"adaptation={adaptation}"])
    validate_experiment_scenario(config)
    assert config["environment"] == scratch["environment"]
    assert config["agents"] == scratch["agents"]
    values = config["training_defaults"]
    assert values["require_pretrained_actor"] == (adaptation != "scratch")
    assert (values.get("lora") is not None) == (adaptation == "lora")
    assert config["mappo"]["actor_mode"] == ("shared" if adaptation == "lora" else "independent")


@pytest.mark.parametrize("path,override,message", [
    (TEAM, "algorithm=ppo", "PPO completion experiments require one vehicle"),
    (PRETRAIN, "algorithm=mappo", "Unsupported evaluation selection strategy"),
    (PRETRAIN, "adaptation=lora", "LoRA requires MAPPO"),
    (PRETRAIN, "network.architecture=cnn", "Only the MLP"),
    (PRETRAIN, "network.encoder=cnn", "identity observation encoder"),
    (PRETRAIN, "network.actor.distribution=normal", "tanh_normal"),
    (TEAM, "network.critic.input_type=local_observation", "global_state input"),
    (TEAM, "network.critic.parameter_sharing=independent", "shared_team parameters"),
    (TEAM, "algorithm.optimizer=sgd", "Unsupported loss, optimizer"),
    (TEAM, "adaptation.load_scope=actor_and_critic", "actor_only loading"),
    (TEAM, "network.actor_hidden_dims=[0]", "positive integers"),
    (TEAM, "adaptation.checkpoint=null", "Fine-tuning requires"),
])
def test_unsupported_combinations_fail_clearly(path, override, message):
    config = load_and_expand_scenario(path, overrides=[override])
    if override == "adaptation.checkpoint=null":
        # Null is valid for scratch, but required for full fine-tuning.
        config = load_and_expand_scenario(path, overrides=["adaptation=full_finetune", override])
    with pytest.raises(ScenarioError, match=message):
        validate_experiment_scenario(config)


def test_hydra_merge_list_null_deletion_and_addition_semantics(tmp_path):
    (tmp_path / "base.yaml").write_text("mapping: {left: 1, right: 2}\nitems: [1, 2]\noptional: 3\n")
    (tmp_path / "primary.yaml").write_text("defaults: [base, _self_]\nmapping: {left: 4}\nitems: [7]\n")
    config = load_yaml_config(tmp_path / "primary.yaml")
    assert config["mapping"] == {"left":4, "right":2} and config["items"] == [7]
    config = compose_configuration(tmp_path / "primary.yaml", [
        "mapping={left:8}", "items=[5,6]", "optional=null", "+new_value=true",
    ])
    assert config["mapping"] == {"left":8, "right":2}
    assert config["items"] == [5,6] and config["optional"] is None and config["new_value"]
    assert "optional" not in compose_configuration(tmp_path / "primary.yaml", ["~optional"])
    with pytest.raises(ScenarioError, match="Could not override"):
        load_and_expand_scenario(tmp_path / "primary.yaml", validate=False, overrides=["typo=1"])
    (tmp_path / "old.yaml").write_text("includes: [base.yaml]\n")
    with pytest.raises(ScenarioError, match="includes.*removed"):
        load_yaml_config(tmp_path / "old.yaml")


def test_explicit_consumer_field_overrides_take_precedence_over_group_defaults():
    config = load_and_expand_scenario("scenarios/ppo_lap_completion_transfer.yaml", overrides=[
        "++experiment.checkpoint=../outputs/custom/best.pt", "++training_defaults.learning_rate=0.002",
    ])
    assert config["experiment"]["checkpoint"] == str(PROJECT_ROOT / "outputs/custom/best.pt")
    assert config["training_defaults"]["learning_rate"] == 0.002
    config = load_and_expand_scenario(TEAM, overrides=["++mappo.actor_mode=shared"])
    validate_experiment_scenario(config)
    assert config["mappo"]["actor_mode"] == config["network"]["actor"]["parameter_sharing"] == "shared"


def test_overrides_apply_once_and_dedicated_cli_flags_take_precedence(monkeypatch, capsys):
    from application.cli import main
    import sys

    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", PRETRAIN,
        "--resolve-config", "--seed", "71", "--total-steps", "9", "--max-speed", "5",
        "--set", "experiment.seed=12", "experiment.total_steps=13", "environment.max_speed=7",
        "++experiment.optional=1", "~experiment.optional"])
    main()
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["experiment"]["seed"] == 71 and config["experiment"]["total_steps"] == 9
    assert config["environment"]["max_speed"] == 5
    assert config["environment"]["vehicle_params"]["wheel_actuators"]["wheel_speed_max"] == 100
    assert "optional" not in config["experiment"]


def test_source_map_checkpoint_and_output_paths_ignore_cwd(tmp_path, monkeypatch):
    from core.environment_config import resolve_environment_config

    monkeypatch.chdir(tmp_path)
    path = "scenarios/ppo_lap_completion_transfer.yaml"
    config = load_and_expand_scenario(path, overrides=["maps.root=maps"])
    assert source_path(path) == PROJECT_ROOT / path
    assert config["experiment"]["checkpoint"] == str(PROJECT_ROOT / "outputs/pretrain/best.pt")
    assert config["paths"]["source_root"] == str(PROJECT_ROOT / "src")
    assert config["paths"]["output_root"] == str(PROJECT_ROOT / "outputs")
    env = resolve_environment_config(config, mode="eval", scenario_dir=PROJECT_ROOT / "scenarios")
    assert env["map_dir"] == str(PROJECT_ROOT / "maps")
    assert Path(env["map_dir"], env["map_yaml"]).is_file()
    assert Path.cwd() == tmp_path
    config = load_and_expand_scenario(path, overrides=["adaptation.checkpoint=outputs/source/best.pt"])
    assert config["experiment"]["checkpoint"] == str(PROJECT_ROOT / "outputs/source/best.pt")


@pytest.mark.parametrize("path,flag", [(PRETRAIN, "--checkpoint"), (TEAM, "--pretrained-actor")])
def test_cli_checkpoint_choices_are_present_in_resolved_configuration(path, flag, monkeypatch, capsys):
    from application.cli import main
    import sys

    monkeypatch.setattr(sys, "argv", ["run.py", "--scenario", path, "--resolve-config",
                                      flag, "outputs/source/best.pt"])
    main()
    config = yaml.safe_load(capsys.readouterr().out)
    expected = str(PROJECT_ROOT / "outputs/source/best.pt")
    assert config["adaptation"]["mode"] == "full_finetune"
    assert config["adaptation"]["checkpoint"] == expected
    if flag == "--checkpoint":
        assert config["experiment"]["checkpoint"] == expected
    else:
        assert config["training_defaults"]["pretrained_actor_checkpoint"] == expected
        assert config["training_defaults"]["require_pretrained_actor"]


def test_selection_and_final_maps_and_seeds_are_explicit_and_stable():
    config = load_and_expand_scenario(TEAM, overrides=[
        "maps.train=[circle_map]", "maps.selection=[Budapest_map]", "maps.final_test=[Spa_map]",
    ])
    validate_experiment_scenario(config)
    selection = resolve_evaluation_protocol(config, "selection")
    final = resolve_evaluation_protocol(config, "final")
    assert selection["seed"] == 10042 and selection["episodes"] == 90
    assert final["seed"] == 20042 and final["episodes"] == 180
    assert selection["map_bundles"] == ["Budapest_map"] and final["map_bundles"] == ["Spa_map"]
    assert config["environment"]["map_bundles_train"] == ["circle_map"]
    assert config["environment"]["map_bundles_eval"] == ["Budapest_map"]
    default = load_and_expand_scenario(TEAM)
    assert default["maps"]["train"] == default["maps"]["selection"] == default["maps"]["final_test"] == ["circle_map"]


def test_task_factory_loads_project_relative_scenarios_outside_project(tmp_path, monkeypatch):
    from core.task_builder import create_race_task

    monkeypatch.chdir(tmp_path)
    task = create_race_task(PRETRAIN)
    try:
        snapshot = task.reset(seed=7)
        assert task.episode_metadata.map_id == "circle_map"
        assert snapshot.observations["car_0"].shape == (158,)
        assert task.spec.state_dim == 17
        assert Path.cwd() == tmp_path
    finally:
        task.close()


def test_resolved_config_and_choices_are_saved_with_experiments(tmp_path):
    from loggers.csv_logger import CSVLogger

    config = load_and_expand_scenario(TEAM, overrides=["adaptation=lora", "adaptation.lora.rank=2"])
    logger = CSVLogger(tmp_path, config)
    logger.close()
    snapshot = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert snapshot == config and "${" not in (tmp_path / "resolved_config.yaml").read_text()
    assert snapshot["_configuration"]["choices"]["adaptation"] == "lora"
    assert snapshot["adaptation"]["lora"]["rank"] == 2
    assert json.loads((tmp_path / "config_snapshot.json").read_text())["config"] == config
