#!/usr/bin/env python3

import json


class Target:
    name: str
    os: str
    runner: str
    statetest_runner: str
    arch: str
    modes: list[str]

    def __init__(
        self,
        name: str,
        os: str,
        runner: str,
        statetest_runner: str,
        arch: str,
        modes: list[str],
    ):
        self.name = name
        self.os = os
        self.runner = runner
        self.statetest_runner = statetest_runner
        self.arch = arch
        self.modes = modes


class Case:
    kind: str
    toolchain: str
    profile: str
    mode: str

    def __init__(
        self,
        kind: str,
        toolchain: str = "stable",
        profile: str = "",
        mode: str = "",
    ):
        self.kind = kind
        self.toolchain = toolchain
        self.profile = profile
        self.mode = mode


class Expanded:
    name: str
    kind: str
    os: str
    runner: str
    arch: str
    toolchain: str
    profile: str
    mode: str

    def __init__(self, target: Target, case: Case):
        self.name = name(target, case)
        self.kind = case.kind
        self.os = target.os
        self.runner = runner(target, case)
        self.arch = target.arch
        self.toolchain = case.toolchain
        self.profile = case.profile
        self.mode = case.mode


toolchains = ["stable", "nightly"]
profiles = ["dev", "release"]

targets = [
    Target(
        "ubuntu-x64",
        "ubuntu",
        "depot-ubuntu-latest",
        "depot-ubuntu-latest-8",
        "x64",
        ["interpreter", "jit", "aot"],
    ),
    Target(
        "macos-arm64",
        "macos",
        "macos-latest",
        "depot-macos-latest",
        "arm64",
        ["interpreter", "jit"],
    ),
]


def main():
    expanded = []
    for target in targets:
        for case in target_cases(target):
            expanded.append(vars(Expanded(target, case)))

    print_json({"include": expanded})


def target_cases(target: Target):
    cases = [
        Case(kind="test", toolchain=toolchain, profile=profile)
        for toolchain in toolchains
        for profile in profiles
    ]
    cases.extend(Case(kind="statetest", mode=mode) for mode in target.modes)
    return cases


def name(target: Target, case: Case):
    if case.kind == "test":
        return f"test {target.name} {case.toolchain} {case.profile}"
    return f"statetest {target.name} {case.mode}"


def runner(target: Target, case: Case):
    if case.kind == "statetest":
        return target.statetest_runner
    return target.runner


def print_json(obj):
    print(json.dumps(obj), end="", flush=True)


if __name__ == "__main__":
    main()
