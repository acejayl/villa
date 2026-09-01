"""max_recent must actually keep max_recent checkpoints.

The trainer records each new checkpoint in checkpoint_history - a
deque(maxlen=3) - and then calls manage_checkpoint_history, which appended the
same entry a second time. Two entries per epoch into a three-slot deque means
the history never held more than two distinct epochs, and the prune at the end
of that function deletes any epoch file the history does not name.

So a run configured for three recent checkpoints kept two. Losing the
third-oldest matters exactly when it matters most: a run that diverges or is
killed leaves fewer intact epochs to fall back to than the setting promised.

manage_debug_gifs had the same duplicate.
"""

from collections import deque

from vesuvius.models.training.save_checkpoint import (
    manage_checkpoint_history,
    manage_debug_gifs,
)

MAX_RECENT = 3


def run_epochs(tmp_path, count, losses=None, max_best=2):
    """Drive the helper the way the trainer does, one epoch at a time."""
    history = deque(maxlen=MAX_RECENT)
    best = []
    kept_per_epoch = []
    for epoch in range(count):
        path = tmp_path / f"model_epoch{epoch}.pth"
        path.write_bytes(b"checkpoint")
        history.append((epoch, str(path)))  # the trainer records it first
        history, best = manage_checkpoint_history(
            checkpoint_history=history,
            best_checkpoints=best,
            epoch=epoch,
            checkpoint_path=path,
            validation_loss=losses[epoch] if losses else 1.0 / (epoch + 1),
            checkpoint_dir=tmp_path,
            model_name="model",
            max_recent=MAX_RECENT,
            max_best=max_best,
        )
        kept_per_epoch.append(
            sorted(int(p.stem.split("epoch")[1]) for p in tmp_path.glob("model_epoch*.pth"))
        )
    return kept_per_epoch, history, best


def test_the_last_three_epochs_are_all_kept(tmp_path):
    kept, _, _ = run_epochs(tmp_path, 6)

    # Once enough epochs exist, the three most recent must all be on disk.
    for epoch in range(MAX_RECENT - 1, 6):
        recent = list(range(epoch - MAX_RECENT + 1, epoch + 1))
        assert set(recent) <= set(kept[epoch]), (
            f"after epoch {epoch} the last {MAX_RECENT} checkpoints should be "
            f"{recent}, on disk: {kept[epoch]}"
        )


def test_history_holds_max_recent_distinct_epochs(tmp_path):
    _, history, _ = run_epochs(tmp_path, 6)

    epochs = [epoch for epoch, _ in history]
    assert len(set(epochs)) == MAX_RECENT
    assert len(epochs) == len(set(epochs)), f"history has duplicates: {epochs}"


def test_a_recent_checkpoint_survives_even_with_a_worse_loss(tmp_path):
    """Recent and best are separate guarantees; a bad last epoch is still kept."""
    losses = [0.9, 0.8, 0.7, 0.6, 5.0]  # the final epoch is far worse
    kept, _, _ = run_epochs(tmp_path, 5, losses=losses)

    assert {2, 3, 4} <= set(kept[-1]), f"on disk: {kept[-1]}"


def test_best_checkpoints_are_still_capped(tmp_path):
    _, _, best = run_epochs(tmp_path, 6)

    assert len(best) <= 2
    assert best == sorted(best, key=lambda entry: entry[0])


def test_debug_gifs_keep_the_last_three_too(tmp_path):
    history = deque(maxlen=MAX_RECENT)
    best = []
    for epoch in range(6):
        path = tmp_path / f"model_debug_epoch{epoch}.gif"
        path.write_bytes(b"gif")
        history.append((epoch, str(path)))
        history, best = manage_debug_gifs(
            debug_gif_history=history,
            best_debug_gifs=best,
            epoch=epoch,
            gif_path=path,
            validation_loss=1.0 / (epoch + 1),
            checkpoint_dir=tmp_path,
            model_name="model",
            max_recent=MAX_RECENT,
            max_best=2,
        )

    on_disk = {int(p.stem.split("epoch")[1]) for p in tmp_path.glob("model_debug_epoch*.gif")}
    assert {3, 4, 5} <= on_disk, f"on disk: {sorted(on_disk)}"
