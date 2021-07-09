# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the example CompilerGym service."""
import pickle

import gym
import numpy as np
import pytest

from compiler_gym.envs import GccEnv
from compiler_gym.service import SessionNotFound
from compiler_gym.spaces import Scalar, Sequence
from tests.test_main import main


@pytest.fixture(scope="function")
def env() -> GccEnv:
    """Test fixture that yields an environment."""
    return gym.make("gcc-v0")


def test_versions(env: GccEnv):
    """Tests the GetVersion() RPC endpoint."""
    assert env.compiler_version == "1.0.0"


def test_action_space(env: GccEnv):
    """Test that the environment reports the service's action spaces."""
    assert env.action_spaces[0].name == "default"
    assert len(env.action_spaces[0].names) == 2281
    assert env.action_spaces[0].names[0] == "-O0"


def test_observation_spaces(env: GccEnv):
    """Test that the environment reports the service's observation spaces."""
    env.reset()
    assert env.observation.spaces.keys() == {
        "source",
        "asm",
        "asm-size",
        "asm-hash",
        "obj",
        "obj-size",
        "obj-hash",
        "gcc-spec",
        "choices",
        "command-line",
    }
    assert env.observation.spaces["obj-size"].space == Scalar(
        min=0, max=np.iinfo(np.int64).max, dtype=np.int64
    )
    assert env.observation.spaces["asm"].space == Sequence(
        size_range=(0, None), dtype=str, opaque_data_format=""
    )


def test_reward_spaces(env: GccEnv):
    """Test that the environment reports the service's reward spaces."""
    env.reset()
    assert env.reward.spaces.keys() == {"asm-size", "obj-size"}


def test_step_before_reset(env: GccEnv):
    """Taking a step() before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        env.step(0)


def test_observation_before_reset(env: GccEnv):
    """Taking an observation before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.observation["asm"]


def test_reward_before_reset(env: GccEnv):
    """Taking a reward before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.reward["obj-size"]


def test_reset_invalid_benchmark(env: GccEnv):
    """Test requesting a specific benchmark."""
    with pytest.raises(LookupError) as ctx:
        env.reset(benchmark="example-v0/foobar")
    assert str(ctx.value) == "Unknown program name"


def test_invalid_observation_space(env: GccEnv):
    """Test error handling with invalid observation space."""
    with pytest.raises(LookupError):
        env.observation_space = 100


def test_invalid_reward_space(env: GccEnv):
    """Test error handling with invalid reward space."""
    with pytest.raises(LookupError):
        env.reward_space = 100


def test_double_reset(env: GccEnv):
    """Test that reset() can be called twice."""
    env.reset()
    assert env.in_episode
    env.step(env.action_space.sample())
    env.reset()
    env.step(env.action_space.sample())
    assert env.in_episode


def test_Step_out_of_range(env: GccEnv):
    """Test error handling with an invalid action."""
    env.reset()
    with pytest.raises(ValueError) as ctx:
        env.step(10000)
    assert str(ctx.value) == "Out-of-range"


def test_default_reward(env: GccEnv):
    """Test default reward space."""
    env.reward_space = "obj-size"
    env.reset()
    observation, reward, done, info = env.step(0)
    assert observation is None
    assert reward == 0
    assert not done


def test_source_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["source"][:20] == "/*\n+----------------"


def test_asm_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["asm"][:20] == "\t.text\n\t.globl _tqmf"


def test_asm_size_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["asm-size"] == 44089


def test_asm_hash_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["asm-hash"] == "79a55346c10d6ea019050bfa0d1ab402"


def test_obj_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["obj"][:5] == b"\xcf\xfa\xed\xfe\x07"


def test_obj_size_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["obj-size"] == 14748


def test_obj_hash_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["obj-hash"] == "582614df51c4d7307e117a8331c55e67"


def test_choices_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    choices = env.observation["choices"]
    assert len(choices) == 502
    assert all(map(lambda x: x == -1, choices))


def test_command_line_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    command_line = env.observation["command-line"]
    assert command_line == "gcc-11 -c src.c -o obj.o"


def test_gcc_spec_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    spec = pickle.loads(env.observation["gcc-spec"])
    assert spec.bin == "gcc-11"


def test_rewards(env: GccEnv):
    """Test reward spaces."""
    env.reset()
    assert env.reward["asm-size"] == 0
    assert env.reward["obj-size"] == 0
    env.step(env.action_space.names.index("-O3"))
    assert env.reward["asm-size"] == -17817.0
    assert env.reward["obj-size"] == -5212.0


def test_benchmarks(env: GccEnv):
    assert list(env.datasets.benchmark_uris())[0] == "benchmark://chstone-v0/adpcm"


def test_compile(env: GccEnv):
    env.observation_space = "obj-size"
    observation = env.reset()
    assert observation == 14748
    observation, _, _, _ = env.step(env.action_space.names.index("-O0"))
    assert observation == 14748
    observation, _, _, _ = env.step(env.action_space.names.index("-O3"))
    assert observation == 19960
    observation, _, _, _ = env.step(env.action_space.names.index("-finline"))
    assert observation == 19960


def test_fork(env: GccEnv):
    env.reset()
    env.step(0)
    env.step(1)
    other_env = env.fork()
    try:
        assert env.benchmark == other_env.benchmark
        assert other_env.actions == [0, 1]
    finally:
        other_env.close()


if __name__ == "__main__":
    main()
