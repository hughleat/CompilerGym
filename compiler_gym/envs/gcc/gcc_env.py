# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
import codecs
import pickle
from pathlib import Path
from typing import List, Optional, Union

from compiler_gym.datasets import Benchmark
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.gcc.datasets import get_gcc_datasets
from compiler_gym.service import ConnectionOpts
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ObservationType


class AsmSizeReward(Reward):
    def __init__(self):
        super().__init__(
            id="asm-size",
            observation_spaces=["asm-size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str):
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward


class ObjSizeReward(Reward):
    def __init__(self):
        super().__init__(
            id="obj-size",
            observation_spaces=["obj-size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str):
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward


class GccEnv(CompilerEnv):
    def __init__(
        self,
        *args,
        benchmark: Optional[Union[str, Benchmark]] = None,
        datasets_site_path: Optional[Path] = None,
        gcc_bin: Optional[str] = None,
        connection_settings: Optional[ConnectionOpts] = None,
        **kwargs,
    ):
        connection_settings = connection_settings or ConnectionOpts()
        connection_settings.script_env = {"CC": gcc_bin or "gcc"}
        super().__init__(
            *args,
            **kwargs,
            # Set a default benchmark for use.
            benchmark=benchmark or "chstone-v0/adpcm",
            datasets=list(get_gcc_datasets(site_data_base=datasets_site_path)),
            rewards=[AsmSizeReward(), ObjSizeReward()],
            connection_settings=connection_settings,
        )
        self._spec = None
        self._timeout = None

    def reset(
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        retry_count: int = 0,
    ) -> Optional[ObservationType]:
        observation = super().reset(benchmark, action_space, retry_count)
        if self._timeout:
            self.send_param("timeout", str(self._timeout))
        return observation

    @property
    def timeout(self) -> Optional[int]:
        return self._timeout

    @timeout.setter
    def timeout(self, value: Optional[int]):
        self._timeout = value
        self.send_param("timeout", str(value) if value else "")

    @property
    def gcc_spec(self):
        if not self._spec:
            pickled = self.send_param("gcc-spec", "")
            self._spec = pickle.loads(codecs.decode(pickled.encode(), "base64"))
        return self._spec

    @property
    def source(self) -> str:
        return self.observation["source"]

    @property
    def asm(self) -> str:
        return self.observation["asm"]

    @property
    def asm_size(self) -> int:
        return self.observation["asm-size"]

    @property
    def asm_hash(self) -> str:
        return self.observation["asm-hash"]

    @property
    def obj(self) -> bytes:
        return self.observation["obj"]

    @property
    def obj_size(self) -> int:
        return self.observation["obj-size"]

    @property
    def obj_hash(self) -> str:
        return self.observation["obj-hash"]

    @property
    def command_line(self) -> str:
        return self.observation["command-line"]

    @property
    def choices(self) -> List[int]:
        return self.observation["choices"]

    @choices.setter
    def choices(self, choices: List[int]):
        spec = self.gcc_spec
        assert len(spec.options) == len(choices)
        assert all(-1 <= c < len(spec.options[i]) for i, c in enumerate(choices))
        self.send_param("choices", ",".join(map(str, choices)))
