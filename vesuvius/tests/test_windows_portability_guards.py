"""Guards for two Windows-only failures that ubuntu-only CI cannot see.

1. Several modules printed characters with no cp1252 mapping - U+2713 CHECK
   MARK, U+26A0 WARNING SIGN, Greek letters, arrows. When stdout is a pipe or a
   file rather than a console, Python encodes with the locale codepage, which on
   a default Windows install is cp1252. `vesuvius.blend_logits ... > blend.log`,
   the normal way to run a multi-hour job, died with UnicodeEncodeError after
   the spatial index was built and before a single chunk was blended, having
   written nothing.

2. train_winding_model asked the DataLoader for multiprocessing_context="fork"
   unconditionally. Windows offers only "spawn", so DataLoader raised - after
   initialize_datasets had already built the shared segment cache and patch mmap
   pack, throwing the expensive prep away. Both shipped configs beside it set
   num_workers > 0, so they take that branch.

The scan is deliberately source-level. On Linux every one of these strings
encodes fine, so a runtime assertion would pass while the bug was live.
"""

import ast
from pathlib import Path

import pytest

import vesuvius

PACKAGE_ROOT = Path(vesuvius.__file__).parent
WINDOWS_CONSOLE_ENCODING = "cp1252"


def _unencodable(text):
    bad = []
    for char in text:
        try:
            char.encode(WINDOWS_CONSOLE_ENCODING)
        except UnicodeEncodeError:
            bad.append(char)
    return bad


def _print_literals(path):
    """Every string literal that reaches a print() call in this module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.lineno, sub.value


def test_printed_text_survives_a_windows_codepage():
    """No print() anywhere in the package may carry a character cp1252 lacks."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for lineno, text in _print_literals(path):
            bad = _unencodable(text)
            if bad:
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno} contains {bad!r}"
                )
    assert not offenders, (
        "printed text that cannot be encoded on a default Windows console, so a "
        "redirected run dies with UnicodeEncodeError:\n  " + "\n  ".join(offenders)
    )


def _make_dataloader_module():
    pytest.importorskip("accelerate", reason="train_winding_model imports accelerate")
    from vesuvius.neural_tracing.winding_models import train_winding_model

    return train_winding_model


def _capture_dataloader_kwargs(monkeypatch, start_methods):
    torch = pytest.importorskip("torch")
    module = _make_dataloader_module()

    captured = {}

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(
        torch.multiprocessing, "get_all_start_methods", lambda: start_methods
    )
    module.make_dataloader(dataset=object(), config={}, generator=None, num_workers=4)
    return captured


def test_dataloader_does_not_demand_fork_where_it_is_unavailable(monkeypatch):
    captured = _capture_dataloader_kwargs(monkeypatch, ["spawn"])
    assert "multiprocessing_context" not in captured, (
        "asked for a start method this platform does not provide"
    )


def test_fork_is_still_used_where_it_is_available(monkeypatch):
    captured = _capture_dataloader_kwargs(monkeypatch, ["fork", "spawn"])
    assert captured.get("multiprocessing_context") == "fork"
