"""--eigen-output must not look like it works.

Both structure-tensor CLIs declare

    --eigen-output   "Output path for eigenvectors"

and neither ever reads it. _finalize_structure_tensor_torch takes only
zarr_path and writes first_component/, second_component/, normal/, confidence/
and the optional eigenvectors/eigenvalues into that same store. There is no
output-path parameter anywhere in the call chain.

So passing --eigen-output did not redirect anything - it silently wrote into
the --eigen-input store instead. That is worse than a no-op: the caller
believes their results went somewhere else, and their input store was modified.

Until the writer grows an output path, the flag has to say so.
"""

import ast
import inspect

import pytest

MODULES = [
    "vesuvius.structure_tensor.create_st",
    "vesuvius.structure_tensor.run_create_st",
]


def source_of(module_name):
    import importlib

    return inspect.getsource(importlib.import_module(module_name))


@pytest.mark.parametrize("module_name", MODULES)
def test_the_flag_is_read_at_all(module_name):
    """The regression: it was declared and never referenced."""
    source = source_of(module_name)

    assert "--eigen-output" in source, "flag missing entirely"
    assert "args.eigen_output" in source, (
        "--eigen-output is declared but never read, so it silently does nothing"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_setting_it_is_refused_rather_than_ignored(module_name):
    source = source_of(module_name)

    tree = ast.parse(source)
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "eigen_output"
    ]
    assert guards, "no `if args.eigen_output:` guard"

    # The guard must bail out, not fall through.
    for guard in guards:
        assert any(isinstance(n, ast.Return) for n in ast.walk(guard)), (
            "the guard does not return, so the run would continue anyway"
        )


@pytest.mark.parametrize("module_name", MODULES)
def test_the_help_text_does_not_promise_a_redirect(module_name):
    """It used to read 'Output path for eigenvectors'."""
    source = source_of(module_name)

    start = source.index("'--eigen-output'")
    declaration = source[start : start + 400]

    assert "Output path for eigenvectors'" not in declaration
    assert "eigen-input" in declaration, (
        "the help should say where the outputs actually land"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_eigen_input_is_still_what_gets_written_to(module_name):
    """The behaviour the message describes is the real one."""
    source = source_of(module_name)

    assert "zarr_path=args.eigen_input" in source
