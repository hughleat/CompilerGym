import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from gcc_spec import GccSpec

import compiler_gym


class SearchPoint:
    """A pair of compilation choices and the resulting object file size."""

    def __init__(self, choices: List[int], size: Optional[int]):
        """If size is None then float.inf will be used to make comparisons easier."""
        self.choices = choices
        self.size = size if size is not None else float("inf")

    def better_than(self, other: "SearchPoint") -> bool:
        """Determine if this result is better than the best so far.
        The choices are the list of choices to make.
        The size is the size of object file.
        Smaller size is better.
        If the sizes are the same, then the sums of the choices are used.
        """
        if self.size == other.size:
            return sum(self.choices) < sum(other.choices)
        return self.size < other.size


class Search:
    def __init__(self):
        self.best = SearchPoint(None, None)

    def run(self):
        """Run the search. Should be possible to call this in parallel."""
        raise NotImplementedError()

    def random_choices(self, gcc: GccSpec):
        return [random.randint(-1, len(opt) - 1) for opt in gcc.options]

    def log_pt(self, pt: SearchPoint):
        with open("log.log", "a") as f:
            print(f"{pt.size}: {pt.choices}", file=f)


class RandomSearch(Search):
    def __init__(self, n: int = 100):
        super().__init__()
        self.n = n

    # TODO remove Override for testing
    def random_choices(self, gcc: GccSpec) -> List:
        return [random.randint(-1, min(256, len(opt) - 1)) for opt in gcc.options]

    def run(self):
        env = compiler_gym.make("gcc-v0")
        env.reset()
        env.timeout = 20
        gcc = env.gcc_spec

        while self.n > 0:
            env.reset()
            choices = env.choices = self.random_choices(gcc)
            command_line = env.command_line
            size = env.obj_size
            pt = SearchPoint(choices, size)
            self.log_pt(pt)

            print(self.n, size, command_line)
            if size == -1:
                pass
            elif pt.better_than(self.best):
                self.best = pt

            self.n -= 1


if __name__ == "__main__":
    search = RandomSearch(109)
    num_workers = 10
    executor = ThreadPoolExecutor(num_workers)
    for i in range(num_workers):
        executor.submit(RandomSearch.run, search)

    # print(f"Best found {search.best.size}, {search.best.choices}")
