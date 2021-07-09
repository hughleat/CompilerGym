#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Query a GCC binary for version,  optimisation and param spaces.
The goal of this file is to query the available settings in a GCC compiler so
that they don't have to be hard coded.

The particular binary is given by the GCC_BIN environment variable. If not set,
then 'gcc' will be used.

The main entry point to this file is the 'get_spec' function which returns a
GccSpec object. That object describes the version, options and parameters.

Querying these settings is time consuming, so this file tries to cache the
values in a cache directory (~/.cache/compiler_gym/gcc).

Running this file will print the gcc spec to stdout.
"""
import logging
import math
import os
import pathlib
import pickle
import re
import subprocess
import sys
from typing import List, Optional

# The binary to use. Taken from an environment variable. Defaults to 'gcc'.
gcc_bin = os.getenv("GCC_BIN") or "gcc"

# TODO util/cache_path (runfiles)
# Location of the cache directory
cache_dir = pathlib.Path.home() / ".cache" / "compiler_gym" / "gcc"


class Option:
    """An Option is either a command line optimisation setting or a parameter.
    It is essentially a list of the possible values that can be taken.

    Each item is command line parameter. In GCC, all of these are single
    settings, so only need one string to describe them, rather than a list.
    """

    def __len__(self):
        """Number of available settings. Note that the absence of a value is not
        included in this, it is implicit."""
        raise NotImplementedError()

    def __getitem__(self, key: int) -> str:
        """Get the command line argument associated with an index (key)."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Get the name of this option."""
        raise NotImplementedError()


class GccOOption(Option):
    """This class represents the -O0, -O1, -O2, -O3, -Os, and -Ofast options.
    This class starts with no values, we fill them in with
    '__gcc_parse_optimize'.

    The suffixes to append to '-O' are stored in self.values.
    """

    def __init__(self):
        self.values = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return "-O" + self.values[key]

    def __str__(self) -> str:
        return "-O"

    def __repr__(self) -> str:
        return f"<GccOOption values=[{','.join(self.values)}]>"


class GccFlagOption(Option):
    """An ordinary -f flag. These have two possible settings. For a given flag
    name there are '-f<name>' and '-fno-<name>."""

    def __init__(self, name: str):
        self.name = name

    def __len__(self):
        return 2

    def __getitem__(self, key: int) -> str:
        return f"-f{'' if key == 0 else 'no-'}{self.name}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagOption name={self.name}>"


class GccFlagEnumOption(Option):
    """A flag of style '-f<name>=[val1, val2, ...]'.
    'self.name' holds the name. 'self.values' holds the values."""

    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}={self.values[key]}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagEnumOption name={self.name}, values=[{','.join(self.values)}]>"


class GccFlagIntOption(Option):
    """A flag of style '-f<name>=<integer>' where the integer is between min and
    max."""

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max

    def __len__(self):
        return self.max - self.min + 1

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}={self.min + key}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagIntOption name={self.name}, min={self.min}, max={self.max}>"


class GccFlagAlignOption(Option):
    """Alignment flags. These take several forms. See the GCC documentation."""

    def __init__(self, name: str):
        logging.warning(f"Alignment options not properly handled {name}")
        self.name = name

    def __len__(self):
        return 1

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagAlignOption name={self.name}>"


class GccParamEnumOption(Option):
    """A parameter '--param=<name>=[val1, val2, val3]."""

    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return f"--param={self.name}={self.values[key]}"

    def __str__(self) -> str:
        return f"--param={self.name}"

    def __repr__(self) -> str:
        return (
            f"<GccParamEnumOption name={self.name}, values=[{','.join(self.values)}]>"
        )


class GccParamIntOption(Option):
    """A parameter '--param=<name>=<integer>. where the integer is between min
    and max."""

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max

    def __len__(self):
        return self.max - self.min + 1

    def __getitem__(self, key: int) -> str:
        return f"--param={self.name}={self.min + key}"

    def __str__(self) -> str:
        return f"--param={self.name}"

    def __repr__(self) -> str:
        return f"<GccParamIntOption name={self.name}, min={self.min}, max={self.max}>"


class GccSpec:
    """This class combines all the information about the version and options,"""

    def __init__(self, bin: str, version: str, options: List[Option]):
        self.bin = bin
        self.version = version
        self.options = options

    @property
    def size(self) -> int:
        """Calculate the size of the option space"""
        sz = 1
        # Each option can be applied or not
        for option in self.options:
            sz *= len(option) + 1
        return sz


def _gcc_parse_optimize(gcc_bin: str = gcc_bin):
    """Parse the optimisation help string from the GCC binary to find
    options."""

    logging.info("Parsing GCC optimisation space")

    # Call 'gcc --help=optimize -Q'
    args = [gcc_bin, "--help=optimize", "-Q"]
    result = subprocess.run(args, capture_output=True)
    # TODO Check for errors
    # Split into lines. Ignore the first line.
    out = result.stdout.decode().split("\n")[1:]

    # Regex patterns to match the different options
    O_num_pat = re.compile("-O<number>")
    O_pat = re.compile("-O([a-z]+)")
    flag_align_eq_pat = re.compile("-f(align-[-a-z]+)=")
    flag_pat = re.compile("-f([-a-z0-9]+)")
    flag_enum_pat = re.compile("-f([-a-z0-9]+)=\\[([-A-Za-z_\\|]+)\\]")
    flag_interval_pat = re.compile("-f([-a-z0-9]+)=<([0-9]+),([0-9]+)>")
    flag_number_pat = re.compile("-f([-a-z0-9]+)=<number>")

    # The list of options as it gets built up.
    options = {}

    # Add a -O value
    def add_gcc_o(value: str):
        # -O flag
        name = "O"
        # There are multiple -O flags. We add one value at a time.
        opt = options[name] = options.get(name, GccOOption())
        # There shouldn't be any way to overwrite this with the wrong type.
        assert type(opt) == GccOOption
        opt.values.append(value)

    # Add a flag
    def add_gcc_flag(name: str):
        # Straight flag.
        # If there is something else in its place already (like a flag enum),
        # then we don't overwrite it.  Straight flags always have the lowest
        # priority
        options[name] = options.get(name, GccFlagOption(name))

    # Add an enum flag
    def add_gcc_flag_enum(name: str, values: List[str]):
        # Enum flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagEnumOption(name, values)

    # Add an integer flag
    def add_gcc_flag_int(name: str, min: int, max: int):
        # Int flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagIntOption(name, min, max)

    # Add an align flag
    def add_gcc_flag_align(name: str):
        # Align flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagAlignOption(name)

    # Parse a line from the help output
    def parse_line(line: str):
        # The first bit of the line is the specification
        bits = line.split()
        if not bits:
            return
        spec = bits[0]

        # -O<number>
        m = O_num_pat.fullmatch(spec)
        if m:
            for i in range(4):
                add_gcc_o(str(i))
            return

        # -Ostr
        m = O_pat.fullmatch(spec)
        if m:
            add_gcc_o(m.group(1))
            return

        # -falign-str=
        # These have quite complicated semantics
        m = flag_align_eq_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            add_gcc_flag_align(name)
            return

        # -fflag
        m = flag_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            add_gcc_flag(name)
            return

        # -fflag=[a|b]
        m = flag_enum_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            values = m.group(2).split("|")
            add_gcc_flag_enum(name, values)
            return

        # -fflag=<min,max>
        m = flag_interval_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = int(m.group(2))
            max = int(m.group(3))
            add_gcc_flag_int(name, min, max)
            return

        # -fflag=<number>
        m = flag_number_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = 0
            max = 2 << 31 - 1
            add_gcc_flag_int(name, min, max)
            return

        logging.warning(f"Unknown GCC optimisation flag spec, '{line}'")

    # Parse all the lines
    for line in out:
        parse_line(line.strip())

    # Sort and return
    return list(map(lambda x: x[1], sorted(list(options.items()))))


def _gcc_parse_params(gcc_bin: str = gcc_bin):
    """Parse the param help string from the GCC binary to find
    options."""

    # Pretty much identical to _gcc_parse_optimize
    logging.info("Parsing GCC param space")

    args = [gcc_bin, "--help=param", "-Q"]
    result = subprocess.run(args, capture_output=True)
    # TODO check for errors
    out = result.stdout.decode().split("\n")[1:]

    param_enum_pat = re.compile("--param=([-a-zA-Z0-9]+)=\\[([-A-Za-z_\\|]+)\\]")
    param_interval_pat = re.compile("--param=([-a-zA-Z0-9]+)=<(-?[0-9]+),([0-9]+)>")
    param_number_pat = re.compile("--param=([-a-zA-Z0-9]+)=")

    params = {}

    def add_gcc_param_enum(name: str, values: List[str]):
        # Enum param.
        opt = params.get(name)
        assert not opt
        params[name] = GccParamEnumOption(name, values)

    def add_gcc_param_int(name: str, min: int, max: int):
        # Int flag.
        opt = params.get(name)
        assert not opt
        params[name] = GccParamIntOption(name, min, max)

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    def parse_line(line: str):
        bits = line.split()
        if not bits:
            return

        # TODO(hugh): Not sure what the correct behavior is there.
        if len(bits) <= 1:
            return

        spec = bits[0]
        default = bits[1]

        # --param=name=[a|b]
        m = param_enum_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            values = m.group(2).split("|")
            assert not default or default in values
            add_gcc_param_enum(name, values)
            return

        # --param=name=<min,max>
        m = param_interval_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = int(m.group(2))
            max = int(m.group(3))
            if is_int(default):
                assert not default or min <= int(default) <= max
                add_gcc_param_int(name, min, max)
                return

        # --param=name=
        m = param_number_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = 0
            max = 2 << 31 - 1
            if is_int(default):
                dflt = int(default)
                min = min if dflt >= min else dflt
                add_gcc_param_int(name, min, max)
                return

        logging.warning(f"Unknown GCC param flag spec, '{line}'")

    # breakpoint()
    for line in out:
        parse_line(line.strip())

    return list(map(lambda x: x[1], sorted(list(params.items()))))


def _gcc_get_version(gcc_bin: str = gcc_bin) -> Optional[str]:
    """Get the version string"""

    logging.info(f"Getting GCC version for {gcc_bin}")
    try:
        args = [gcc_bin, "--version"]
        result = subprocess.run(args, capture_output=True)
        version = result.stdout.decode().split("\n")[0]
        logging.info(f"GCC version is {version}")
        return version
    except subprocess.SubprocessError:
        logging.error("Unable to get GCC version")
        return None


def _version_hash(version: str) -> str:
    """Hash the version so we can cache the spec at that name."""
    h = 0
    for c in version:
        h = ord(c) + 31 * h
    return h % (2 << 64)


def get_spec(gcc_bin: str = gcc_bin) -> Optional[GccSpec]:
    # Get the version
    version = _gcc_get_version(gcc_bin)
    if not version:
        # Already logged the problem
        return None

    # See if there is a pickled spec in the cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    # First we use a hash to name the file
    spec_filename = f"gcc-spec-{_version_hash(version)}.pkl"
    spec_path = cache_dir / spec_filename

    # Try to get the pickled version
    if os.path.isfile(spec_path):
        # Pickle exists
        with open(spec_path, "rb") as f:
            spec = pickle.load(f)
        spec.gcc_bin = gcc_bin
        logging.info(f"GccSpec for version '{version}' read from {spec_path}")
    else:
        # Pickle doesn't exist, parse
        optim_opts = _gcc_parse_optimize(gcc_bin)
        param_opts = _gcc_parse_params(gcc_bin)
        spec = GccSpec(gcc_bin, version, optim_opts + param_opts)
        if not spec.options:
            return None
        with open(spec_path, "wb") as f:
            pickle.dump(spec, f)
        logging.info(f"GccSpec for version '{version}' written to {spec_path}")

    logging.info(f"GccSpec size is approximately 10^{math.log(spec.size)}")
    return spec


if __name__ == "__main__":
    """Find the spec for GCC and print what is found.
    If an argument is given, use it as the gcc binary path."""
    gcc_bin = "gcc-11"
    spec = get_spec(gcc_bin if len(sys.argv) == 1 else sys.argv[1])

    print(f"GCC Version: {spec.version}")
    print("Options:")
    for option in spec.options:
        print(option.__repr__())
