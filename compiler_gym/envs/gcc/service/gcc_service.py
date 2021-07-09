#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A CompilerGym service for GCC.
This service reads a specification of the compiler from the binary using the
code in 'gcc_spec.py'. To change the compiler from the default 'gcc', set the
'GCC_BIN' environment variable.
"""
import hashlib
import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from compiler_gym.envs.gcc.service import gcc_spec
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import Action as ProtoAction
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)


class GccCompilationSession(CompilationSession):
    """A GCC interactive compilation session."""

    compiler_version: str = "1.0.0"

    """The GCCSpec for this compiler"""
    spec = gcc_spec.get_spec()

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        # The benchmark being used
        self.benchmark = benchmark
        if self.spec:
            # The current choices for each options. '-1' indicates the implicit
            # missing option.
            self.choices = [-1] * len(self.spec.options)
            # The source code
            self._source = None
            # The assembled code
            self._asm = None
            # Size of the assembled code
            self._asm_size = None
            # Hash of the assembled code
            self._asm_hash = None
            # The object binary
            self._obj = None
            # size of the object binary
            self._obj_size = None
            # Hash of the object binary
            self._obj_hash = None

            logging.info("Started a compilation session for %s", benchmark.uri)
        else:
            raise RuntimeError("Unable to create GCC spec, check the GCC_BIN env var.")

    @property
    def source(self) -> str:
        """Get the benchmark source"""
        self.prepare_files()
        return self._source

    @property
    def asm(self) -> bytes:
        """Get the assembled code"""
        self.assemble()
        return self._asm

    @property
    def asm_size(self) -> int:
        """Get the assembled code size"""
        self.assemble()
        return self._asm_size

    @property
    def asm_hash(self) -> str:
        """Get the assembled code hash"""
        self.assemble()
        return self._asm_hash

    @property
    def obj(self) -> bytes:
        """Get the compiled code"""
        self.compile()
        return self._obj

    @property
    def obj_size(self) -> int:
        """Get the compiled code size"""
        self.compile()
        return self._obj_size

    @property
    def obj_hash(self) -> str:
        """Get the compiled code hash"""
        self.compile()
        return self._obj_hash

    @property
    def gcc_spec(self) -> bytes:
        """Get the pickled spec"""
        return pickle.dumps(self.spec)

    @property
    def src_path(self) -> Path:
        """Get the path to the source file"""
        return self.working_dir / "src.c"

    @property
    def obj_path(self) -> Path:
        """Get the path to object file"""
        return self.working_dir / "obj.o"

    @property
    def asm_path(self) -> Path:
        """Get the path to the assembly"""
        return self.working_dir / "asm.s"

    def obj_command_line(
        self, src_path: Path = None, obj_path: Path = None
    ) -> List[str]:
        """Get the command line to create the object file.
        The 'src_path' and 'obj_path' give the input and output paths. If not
        set, then they are taken from 'self.src_path' and 'self.obj_path'. This
        is useful for printing where the actual paths are not important."""
        src_path = src_path or self.src_path
        obj_path = obj_path or self.obj_path
        # Gather the choices as strings
        opts = [
            option[choice]
            for option, choice in zip(self.spec.options, self.choices)
            if choice >= 0
        ]
        cmd_line = [self.spec.bin] + opts + ["-c", src_path, "-o", obj_path]
        return cmd_line

    def asm_command_line(
        self, src_path: Path = None, asm_path: Path = None
    ) -> List[str]:
        """Get the command line to create the assembly file.
        The 'src_path' and 'obj_path' give the input and output paths. If not
        set, then they are taken from 'self.src_path' and 'self.obj_path'. This
        is useful for printing where the actual paths are not important."""
        src_path = src_path or self.src_path
        asm_path = asm_path or self.asm_path
        opts = [
            option[choice]
            for option, choice in zip(self.spec.options, self.choices)
            if choice >= 0
        ]
        cmd_line = [self.spec.bin] + opts + ["-S", src_path, "-o", asm_path]
        return cmd_line

    def prepare_files(self):
        """Copy the source to the working directory."""
        if not self._source:
            if self.benchmark.program.contents:
                with open(self.src_path, "w") as f:
                    print(self.benchmark.program.contents.decode(), file=f)
                self._source = self.benchmark.program.contents.decode()
            else:
                raise NotImplementedError("Program with URI")

    def compile(self) -> Optional[str]:
        """Compile the benchmark"""
        if not self._obj:
            self.prepare_files()
            logging.info(f"Compiling: {' '.join(map(str, self.obj_command_line()))}")
            subprocess.run(self.obj_command_line(), cwd=self.working_dir)
            with open(self.obj_path, "rb") as f:
                # Set the internal variables
                self._obj = f.read()
                self._obj_size = os.path.getsize(self.obj_path)
                self._obj_hash = hashlib.md5(self._obj).hexdigest()

    def assemble(self) -> Optional[str]:
        """Assemble the benchmark"""
        if not self._obj:
            self.prepare_files()
            logging.info(f"Assembling: {' '.join(map(str, self.asm_command_line()))}")
            subprocess.run(self.asm_command_line(), cwd=self.working_dir)
            logging.info("Assembled")
            with open(self.asm_path, "rb") as f:
                # Set the internal variables
                asm_bytes = f.read()
                self._asm = asm_bytes.decode()
                self._asm_size = os.path.getsize(self.asm_path)
                self._asm_hash = hashlib.md5(asm_bytes).hexdigest()

    def apply_action(
        self, proto_action: ProtoAction
    ) -> Tuple[bool, Optional[ActionSpace], bool]:
        """Apply an action."""
        if proto_action.action < 0 or proto_action.action > len(
            self.action_spaces[0].action
        ):
            raise ValueError("Out-of-range")

        # Get the action
        action = self.actions[proto_action.action]
        # Apply the action to this session and check if we changed anything
        old_choices = self.choices.copy()
        action(self)
        logging.info("Applied action " + str(action))

        # Reset the internal variables if this action has caused a change in the
        # choices
        if old_choices != self.choices:
            self._obj = None
            self._obj_size = None
            self._obj_hash = None
            self._asm = None
            self._asm_size = None
            self._asm_hash = None

        return False, None, False

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        """Get one of the observations"""
        if observation_space.name == "source":
            return Observation(string_value=self.source)
        elif observation_space.name == "asm":
            return Observation(string_value=self.asm)
        elif observation_space.name == "asm-size":
            return Observation(scalar_int64=self.asm_size)
        elif observation_space.name == "asm-hash":
            return Observation(string_value=self.asm_hash)
        elif observation_space.name == "obj":
            return Observation(binary_value=self.obj)
        elif observation_space.name == "obj-size":
            return Observation(scalar_int64=self.obj_size)
        elif observation_space.name == "obj-hash":
            return Observation(string_value=self.obj_hash)
        elif observation_space.name == "choices":
            observation = Observation()
            observation.int64_list.value[:] = self.choices
            return observation
        elif observation_space.name == "command-line":
            return Observation(
                string_value=" ".join(map(str, self.obj_command_line("src.c", "obj.o")))
            )
        elif observation_space.name == "gcc-spec":
            return Observation(binary_value=self.gcc_spec)
        elif observation_space.name == "features":
            observation = Observation()
            observation.int64_list.value[:] = [0, 0, 0]
            return observation
        elif observation_space.name == "runtime":
            return Observation(scalar_double=0)
        else:
            raise KeyError(observation_space.name)


class Action:
    """An action is applying a choice to an option"""

    def __init__(self, option: gcc_spec.Option, option_index: int):
        """The option and its index in the option list.  We need the index to
        match it with the corresponding choice later during the application of
        the action."""
        self.option = option
        self.option_index = option_index

    def __call__(self, session: GccCompilationSession):
        """Apply the action to the session."""
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class SimpleAction(Action):
    """A simple action just sets the choice directly.
    The choice_index describes which choice to apply."""

    def __init__(self, option: gcc_spec.Option, option_index: int, choice_index: int):
        super().__init__(option, option_index)
        self.choice_index = choice_index

    def __call__(self, session: GccCompilationSession):
        session.choices[self.option_index] = self.choice_index

    def __str__(self) -> str:
        return self.option[self.choice_index]


class IncrAction(Action):
    """An action that increments a choice by an amount."""

    def __init__(self, option: gcc_spec.Option, option_index: int, choice_incr: int):
        super().__init__(option, option_index)
        self.choice_incr = choice_incr

    def __call__(self, session: GccCompilationSession):
        choice = session.choices[self.option_index]
        choice += self.choice_incr
        if choice < -1:
            choice = -1
        if choice >= len(self.option):
            choice = len(self.option) - 1
        session.choices[self.option_index] = choice

    def __str__(self) -> str:
        return f"{self.option}[{self.choice_incr:+}]"


if GccCompilationSession.spec:
    # The available actions
    GccCompilationSession.actions = []

    # Actions that are small will have all their various choices made as
    # explicit actions.
    # Actions that are not small will have the abbility to increment the choice
    # by different amounts.
    for i, option in enumerate(GccCompilationSession.spec.options):
        if len(option) < 10:
            for j in range(len(option)):
                GccCompilationSession.actions.append(SimpleAction(option, i, j))
        if len(option) >= 10:
            GccCompilationSession.actions.append(IncrAction(option, i, 1))
            GccCompilationSession.actions.append(IncrAction(option, i, -1))
        if len(option) >= 50:
            GccCompilationSession.actions.append(IncrAction(option, i, 10))
            GccCompilationSession.actions.append(IncrAction(option, i, -10))
        if len(option) >= 500:
            GccCompilationSession.actions.append(IncrAction(option, i, 100))
            GccCompilationSession.actions.append(IncrAction(option, i, -100))
        if len(option) >= 5000:
            GccCompilationSession.actions.append(IncrAction(option, i, 1000))
            GccCompilationSession.actions.append(IncrAction(option, i, -1000))

    # The action spaces. Just wraps the 'actions' list.
    GccCompilationSession.action_spaces = [
        ActionSpace(
            name="default", action=list(map(str, GccCompilationSession.actions))
        )
    ]

    # A list of observation spaces supported by this service.
    GccCompilationSession.observation_spaces = [
        # A string of the source code
        ObservationSpace(
            name="source",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=False,
            default_value=Observation(string_value=""),
        ),
        # A string of the assembled code
        ObservationSpace(
            name="asm",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(string_value=""),
        ),
        # The size of the assembled code
        ObservationSpace(
            name="asm-size",
            scalar_int64_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
        # The hash of the assembled code
        ObservationSpace(
            name="asm-hash",
            string_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=200)
            ),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(string_value=""),
        ),
        # A bytes of the object code
        ObservationSpace(
            name="obj",
            binary_size_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=False,
            default_value=Observation(binary_value=b""),
        ),
        # The size of the object code
        ObservationSpace(
            name="obj-size",
            scalar_int64_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
        # The hash of the object code
        ObservationSpace(
            name="obj-hash",
            string_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=200)
            ),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(string_value=""),
        ),
        # A list of the choices. Each element corresponds to an option in the spec.
        # '-1' indicates that this is empty on the command line (e.g. if the choice
        # corresponding to the '-O' option is -1, then no -O flag will be emitted.)
        # If a nonnegative number if given then that particular choice is used
        # (e.g. for the -O flag, 5 means use '-Ofast' on the command line.)
        ObservationSpace(
            name="choices",
            int64_range_list=ScalarRangeList(
                range=[
                    ScalarRange(
                        min=ScalarLimit(value=0), max=ScalarLimit(value=len(option) - 1)
                    )
                    for option in GccCompilationSession.spec.options
                ]
            ),
        ),
        # The command line for compiling the object file as a string
        ObservationSpace(
            name="command-line",
            string_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=200)
            ),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(string_value=""),
        ),
        # The pickled spec of the compiler
        ObservationSpace(
            name="gcc-spec",
            binary_size_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=True,
            default_value=Observation(binary_value=b""),
        ),
    ]

else:
    raise RuntimeError(
        "Unable to create GCC spec.\n"
        + f" Is the GCC_BIN ({os.getenv('GCC_BIN')}) environment set correctly?"
    )
