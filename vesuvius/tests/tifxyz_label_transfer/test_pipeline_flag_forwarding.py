"""Flags the pipeline subcommand accepts must reach its stages.

run_pipeline runs two single transfers, building each stage's namespace from
scratch in _single_namespace_from_pipeline. Any option that both subcommands
accept but that function does not copy across is dropped - and because the
stages read these with getattr defaults rather than attribute access, dropping
one is silent. --rasterizer fell back to "auto" and --no-stream-output to off,
whatever the command line said.

This test derives the list of shared options from the parser itself, so a new
one added to the shared group fails here rather than quietly doing nothing.
"""

import argparse
import inspect
import re

import pytest

from vesuvius.tifxyz_label_transfer import transfer


def shared_option_dests():
    parser = transfer.build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ][0]
    single = {a.dest for a in subparsers.choices["single"]._actions}
    pipeline = {a.dest for a in subparsers.choices["pipeline"]._actions}
    return (single & pipeline) - {"help"}


def forwarded_names():
    source = inspect.getsource(transfer._single_namespace_from_pipeline)
    return set(re.findall(r"^\s{8}(\w+)=", source, re.M))


def test_every_shared_option_is_forwarded_to_the_stages():
    missing = sorted(shared_option_dests() - forwarded_names())

    assert not missing, (
        "options accepted by both subcommands but dropped when the pipeline "
        f"builds a stage namespace, so they silently do nothing: {missing}"
    )


@pytest.mark.parametrize(
    "flag,value,attribute",
    [
        (["--rasterizer", "python"], "python", "rasterizer"),
        (["--no-stream-output"], True, "no_stream_output"),
    ],
)
def test_the_value_on_the_command_line_reaches_the_stage(flag, value, attribute):
    parser = transfer.build_parser()
    args = parser.parse_args([
        "pipeline",
        "--old-tifxyz", "old",
        "--updated-tifxyz", "updated",
        "--target-tifxyz", "target",
        "--affine", "affine.json",
        "--label", "label.tif",
        "--output", "out.tif",
        *flag,
    ])

    stage = transfer._single_namespace_from_pipeline(
        args,
        source_tifxyz="old",
        target_tifxyz="updated",
        label="label.tif",
        output="out.tif",
        affine=None,
        affine_direction="forward",
        max_distance=None,
        target_reference=None,
        output_shape=None,
    )

    assert getattr(stage, attribute) == value


def test_the_defaults_survive_the_hop():
    parser = transfer.build_parser()
    args = parser.parse_args([
        "pipeline",
        "--old-tifxyz", "old",
        "--updated-tifxyz", "updated",
        "--target-tifxyz", "target",
        "--affine", "affine.json",
        "--label", "label.tif",
        "--output", "out.tif",
    ])

    stage = transfer._single_namespace_from_pipeline(
        args,
        source_tifxyz="old",
        target_tifxyz="updated",
        label="label.tif",
        output="out.tif",
        affine=None,
        affine_direction="forward",
        max_distance=None,
        target_reference=None,
        output_shape=None,
    )

    assert stage.rasterizer == args.rasterizer
    assert stage.no_stream_output == args.no_stream_output
