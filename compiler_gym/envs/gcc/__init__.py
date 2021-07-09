# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from pathlib import Path
from typing import Iterable, Optional, Union

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.envs.compiler_env import CompilerEnv

# from compiler_gym.envs.gcc.datasets import get_gcc_datasets
from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

GCC_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/gcc/service/compiler_gym-gcc-service"
)


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


foo_c = """
#include "stdio.h"

int main(int argc, char* argv[]) {
    printf("Hello, World\\n");
    return 0;
}
"""


class ExampleDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://example-v0",
            license="MIT",
            description="An example dataset",
            site_data_base=site_data_path("example_dataset"),
        )
        self._benchmarks = {
            "benchmark://example-v0/foo": Benchmark.from_file_contents(
                "benchmark://example-v0/foo", foo_c.encode("utf-8")
            ),
            "benchmark://example-v0/bar": Benchmark.from_file_contents(
                "benchmark://example-v0/bar", "Ir data".encode("utf-8")
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")


class GccEnv(CompilerEnv):
    def __init__(
        self,
        *args,
        benchmark: Optional[Union[str, Benchmark]] = None,
        datasets_site_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            # Set a default benchmark for use.
            benchmark=benchmark or "cbench-v1/qsort",
            datasets=[
                ExampleDataset()
            ],  # get_gcc_datasets(site_data_base=datasets_site_path),
            rewards=[AsmSizeReward(), ObjSizeReward()],
        )


register(
    id="gcc-v0",
    entry_point="compiler_gym.envs.gcc:GccEnv",
    kwargs={
        "service": GCC_SERVICE_BINARY,
        # "rewards": [AsmSizeReward(), ObjSizeReward()],
        # "datasets": [ExampleDataset()],
    },
)
