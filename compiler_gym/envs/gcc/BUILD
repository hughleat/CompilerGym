# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "gcc",
    srcs = ["__init__.py"],
    data = [
        "//compiler_gym/envs/gcc/service:compiler_gym-gcc-service",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/util",
    ],
)

py_test(
    name = "gcc_tests",
    srcs = ["gcc_tests.py"],
    deps = [
        ":gcc",
        "//compiler_gym",
        "//tests:test_main",
    ],
)
