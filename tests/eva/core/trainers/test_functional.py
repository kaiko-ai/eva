"""Tests for trainer session functions."""

from unittest import mock

from eva.core.trainers import functional


def test_run_evaluation_session_combines_results_per_dataloader_pair() -> None:
    """Tests that validation and test results are combined by dataloader index."""
    trainer = mock.Mock(default_log_dir="logs")
    model = mock.Mock()
    datamodule = mock.Mock()
    recorder = mock.Mock()
    validation_scores = [
        {"val/MulticlassAccuracy": 0.1},
        {"val/MulticlassAccuracy": 0.2},
    ]
    test_scores = [
        {"test/MulticlassAccuracy": 0.3},
        {"test/MulticlassAccuracy": 0.4},
    ]

    with (
        mock.patch.object(functional._recorder, "SessionRecorder", return_value=recorder),
        mock.patch.object(
            functional,
            "run_evaluation",
            return_value=(validation_scores, test_scores),
        ),
    ):
        functional.run_evaluation_session(
            base_trainer=trainer,
            base_model=model,
            datamodule=datamodule,
            combine_dataloader_results=True,
            verbose=False,
        )

    assert recorder.update.call_args_list == [
        mock.call([validation_scores[0]], [test_scores[0]]),
        mock.call([validation_scores[1]], [test_scores[1]]),
    ]
    recorder.save.assert_called_once_with()


def test_run_evaluation_session_combines_validate_only_results() -> None:
    """Tests that validate-only runs do not pass placeholder test results."""
    trainer = mock.Mock(default_log_dir="logs")
    model = mock.Mock()
    datamodule = mock.Mock()
    recorder = mock.Mock()
    validation_scores = [
        {"val/MulticlassAccuracy": 0.1},
        {"val/MulticlassAccuracy": 0.2},
    ]

    with (
        mock.patch.object(functional._recorder, "SessionRecorder", return_value=recorder),
        mock.patch.object(
            functional,
            "run_evaluation",
            return_value=(validation_scores, None),
        ),
    ):
        functional.run_evaluation_session(
            base_trainer=trainer,
            base_model=model,
            datamodule=datamodule,
            combine_dataloader_results=True,
            verbose=False,
        )

    assert recorder.update.call_args_list == [
        mock.call([validation_scores[0]], None),
        mock.call([validation_scores[1]], None),
    ]
    assert mock.call([validation_scores[0]], [None]) not in recorder.update.call_args_list
    assert mock.call([validation_scores[1]], [None]) not in recorder.update.call_args_list
    recorder.save.assert_called_once_with()


def test_run_evaluation_session_keeps_results_separate_when_combine_disabled() -> None:
    """Tests that full result lists are forwarded unchanged when combining is disabled."""
    trainer = mock.Mock(default_log_dir="logs")
    model = mock.Mock()
    datamodule = mock.Mock()
    recorder = mock.Mock()
    validation_scores = [
        {"val/MulticlassAccuracy": 0.1},
        {"val/MulticlassAccuracy": 0.2},
    ]
    test_scores = [
        {"test/MulticlassAccuracy": 0.3},
        {"test/MulticlassAccuracy": 0.4},
    ]

    with (
        mock.patch.object(functional._recorder, "SessionRecorder", return_value=recorder),
        mock.patch.object(
            functional,
            "run_evaluation",
            return_value=(validation_scores, test_scores),
        ),
    ):
        functional.run_evaluation_session(
            base_trainer=trainer,
            base_model=model,
            datamodule=datamodule,
            combine_dataloader_results=False,
            verbose=False,
        )

    recorder.update.assert_called_once_with(validation_scores, test_scores)
    recorder.save.assert_called_once_with()
