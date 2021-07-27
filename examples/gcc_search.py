import random
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import List, Optional, Tuple
import threading

import compiler_gym
from compiler_gym.envs.gcc.gcc_spec import GccSpec


class ChoicesSearchPoint:
    """A pair of compilation choices and the resulting object file size."""

    def __init__(self, choices: List[int], size: Optional[int]):
        """If size is None then float.inf will be used to make comparisons easier."""
        self.choices = choices
        self.size = size if size is not None else float("inf")

    def better_than(self, other: "ChoicesSearchPoint") -> bool:
        """Determine if this result is better than the best so far.
        The choices are the list of choices to make.
        The size is the size of object file.
        Smaller size is better.
        If the sizes are the same, then the sums of the choices are used.
        """
        if self.size == other.size:
            return sum(self.choices) < sum(other.choices)
        return self.size < other.size


class ChoicesSearch:
    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        self.best = ChoicesSearchPoint(None, None)
        
        env = compiler_gym.make("gcc-v0")
        env.reset(benchmark=self.benchmark)
        env.timeout = 20
        env.step(env.action_space.names.index("-Os"))
        self.baseline = ChoicesSearchPoint(env.choices, env.obj_size)
        env.close()
        

    def run(self):
        """Run the search. Should be possible to call this in parallel."""
        raise NotImplementedError()

    def random_choices(self, gcc: GccSpec):
        return [random.randint(-1, len(opt) - 1) for opt in gcc.options]

    def log_pt(self, n: int, pt: ChoicesSearchPoint, env):
        bname = self.benchmark.replace("benchmark://", "")
        fname = "gcc_" + bname.replace('/', '_') + ".log"
        
        with open(fname, "a") as f:
            scale = self.baseline.size / pt.size if pt.size != 0 else "-"
            print(f"{scale}, {pt.size}, {n}, {','.join(map(str, pt.choices))}, {env.command_line}", file=f)

        print(f"{bname} scale={scale}, size={pt.size}, n={n}, choices={','.join(map(str, pt.choices))}, cmdline={env.command_line}")


class RandomChoicesSearch(ChoicesSearch):
    def __init__(self, benchmark: str, n: int = 100):
        super().__init__(benchmark)
        self.n = n

    def random_choices(self, gcc: GccSpec) -> List:
        return [random.randint(-1, min(256, len(opt) - 1)) for opt in gcc.options]

    def run(self):
        env = compiler_gym.make("gcc-v0")
        env.reset(benchmark=self.benchmark)
        env.timeout = 20
        gcc = env.gcc_spec

        while self.n > 0:
            n = self.n
            self.n -= 1
        
            env.reset(benchmark=self.benchmark)
            choices = env.choices = self.random_choices(gcc)
            size = env.obj_size
            
            pt = ChoicesSearchPoint(choices, size)
            self.log_pt(n, pt, env)

            print(n, size, env.command_line)
            if size != -1 and pt.better_than(self.best):
                self.best = pt
        env.close()


class RandomWalkActionsSearch(ChoicesSearch):
    def __init__(self, benchmark: str, n: int = 100, steps: int = 1):
        super().__init__(benchmark)
        self.n = n
        self.steps = steps
    
    def run(self):
        env = compiler_gym.make("gcc-v0")
        env.reset(benchmark=self.benchmark)
        env.timeout = 20
        gcc = env.gcc_spec

        choices = env.choices

        while self.n > 0:
            n = self.n
            self.n -= 1
        
            env.reset(benchmark=self.benchmark)
            env.choices = choices
            for i in range(self.steps):
                env.step(env.action_space.sample())
            size = env.obj_size
            if size != -1:
                choices = env.choices
            
            pt = ChoicesSearchPoint(choices, size)
            self.log_pt(n, pt, env)

            print(n, size, env.command_line)
            if size != -1 and pt.better_than(self.best):
                self.best = pt
        env.close()


class HillClimbActionsSearch(ChoicesSearch):
    def __init__(self, benchmark: str, n: int = 100, steps: int = 1):
        super().__init__(benchmark)
        self.n = n
        self.steps = steps
        self.best = self.baseline
        self.lock = threading.Lock()
    
    def run(self):
        env = compiler_gym.make("gcc-v0")
        env.reset(benchmark=self.benchmark)
        env.timeout = 20
        gcc = env.gcc_spec

        while self.n > 0:
            n = self.n
            self.n -= 1
        
            env.reset(benchmark=self.benchmark)
            env.choices = self.best.choices
            for i in range(self.steps):
                env.step(env.action_space.sample())
            size = env.obj_size
            
            pt = ChoicesSearchPoint(env.choices, size)
            self.log_pt(n, pt, env)

            if size != -1 and pt.better_than(self.best):
                self.best = pt
        env.close()


class GAChoicesSeaerch(ChoicesSearch):
    def __init__(self, benchmark: str, n: int = 10000, pop: int = 100):
        super().__init__(benchmark)
        self.n = n
        self.pop = pop

    def tournament(self, pop: List[ChoicesSearchPoint], k: int = 7) -> ChoicesSearchPoint:
        cands = random.sample(pop, k)
        
        def key(pt: ChoicesSearchPoint) -> Tuple[int, int]:
            s = sum(pt.choices)
            return (pt.size, s) if pt.size != -1 else (float("inf"), s)
            
        return min(cands, key=key)

    def xover(self, a: List[int], b: List[int]) -> List[int]:
        assert(len(a) == len(b))
        
        def select(x: int, y: int) -> int:
            return x if bool(random.getrandbits(1)) else y
                
        c = [select(x, y) for x, y in zip(a, b)]
        return c

    def mutate(self, a: List[int], k: int, gcc: GccSpec) -> List[int]:
        pass
    
    def run(self):
        env = compiler_gym.make("gcc-v0")
        env.reset(benchmark=self.benchmark)
        env.timeout = 20
        gcc = env.gcc_spec

        while self.n > 0:
            n = self.n
            self.n -= 1
        
            env.reset(benchmark=self.benchmark)
            env.choices = self.best.choices
            for i in range(self.steps):
                env.step(env.action_space.sample())
            size = env.obj_size
            
            pt = ChoicesSearchPoint(env.choices, size)
            self.log_pt(pt)

            print(n, size, env.command_line)
            if size != -1 and pt.better_than(self.best):
                self.best = pt
        env.close()


def main():
    search_cls = HillClimbActionsSearch
    kwargs = {"n": 1000, "steps": 10}
    num_workers = 1

    def get_benchmarks():
        benchmarks = []
        env = compiler_gym.make("gcc-v0")
        env.reset()
        for dataset in env.datasets:
            benchmarks += islice(dataset.benchmark_uris(), 50)
        env.close()
        benchmarks.sort()
        return benchmarks

    if len(sys.argv) == 1:
        print("Benchmark not given")
        print("Select from:")
        print('\n'.join(get_benchmarks()))
        return

    if sys.argv[1] == "all":
        benchmarks = get_benchmarks()
    else:
        benchmarks = sys.argv[1:]

    searches = [search_cls(benchmark=benchmark, **kwargs) for benchmark in benchmarks]
    for search in searches:
        for i in range(num_workers):
            search.run() # 3:00.13
            #print(f"*** Submit")
            #threading.Thread(target=search.run).start()


if __name__ == "__main__":
    main()