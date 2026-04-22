"""Tests for trainer session functions."""

from unittest import mock

from eva.core.trainers import functional


def test_run_evaluation_session_records_each_dataset_pair_as_run() -> None:
    """Tests that each validation/test dataset pair is recorded as its own run."""
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
            record_datasets_as_runs=True,
            verbose=False,
        )

    assert recorder.update.call_args_list == [
        mock.call([validation_scores[0]], [test_scores[0]]),
        mock.call([validation_scores[1]], [test_scores[1]]),
    ]
    recorder.save.assert_called_once_with()


def test_run_evaluation_session_records_validate_only_datasets_as_runs() -> None:
    """Tests that validate-only dataset outputs are recorded without placeholder test results."""
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
            record_datasets_as_runs=True,
            verbose=False,
        )

    assert recorder.update.call_args_list == [
        mock.call([validation_scores[0]], None),
        mock.call([validation_scores[1]], None),
    ]
    assert mock.call([validation_scores[0]], [None]) not in recorder.update.call_args_list
    assert mock.call([validation_scores[1]], [None]) not in recorder.update.call_args_list
    recorder.save.assert_called_once_with()


def test_evaluate_stage_calls_validate_per_dataloader_when_record_datasets_as_runs() -> None:
    """Tests that each dataloader is evaluated separately when record_datasets_as_runs is True."""
    trainer = mock.Mock()
    trainer.validate.side_effect = [
        [{"val/Accuracy": 0.8}],
        [{"val/Accuracy": 0.6}],
    ]
    trainer.checkpoint_type = "best"
    model = mock.Mock()
    datamodule = mock.Mock()
    dl_1, dl_2 = mock.Mock(), mock.Mock()
    datamodule.val_dataloader.return_value = [dl_1, dl_2]

    result = functional._evaluate_stage(
        trainer,
        model,
        datamodule,
        stage="validate",
        record_datasets_as_runs=True,
    )

    assert result == [{"val/Accuracy": 0.8}, {"val/Accuracy": 0.6}]
    assert trainer.validate.call_count == 2
    trainer.validate.assert_any_call(
        model=model, dataloaders=[dl_1], verbose=True, ckpt_path="best"
    )
    trainer.validate.assert_any_call(
        model=model, dataloaders=[dl_2], verbose=True, ckpt_path="best"
    )
    datamodule.setup.assert_called_once_with("validate")


def test_evaluate_stage_uses_datamodule_when_single_dataloader() -> None:
    """Tests that a single dataloader falls back to the standard datamodule-based call."""
    trainer = mock.Mock()
    trainer.validate.return_value = [{"val/Accuracy": 0.9}]
    trainer.checkpoint_type = "best"
    model = mock.Mock()
    datamodule = mock.Mock()
    datamodule.val_dataloader.return_value = [mock.Mock()]

    result = functional._evaluate_stage(
        trainer,
        model,
        datamodule,
        stage="validate",
        record_datasets_as_runs=True,
    )

    assert result == [{"val/Accuracy": 0.9}]
    trainer.validate.assert_called_once_with(
        model=model, datamodule=datamodule, verbose=True, ckpt_path="best"
    )


def test_evaluate_stage_uses_datamodule_when_record_disabled() -> None:
    """Tests that record_datasets_as_runs=False uses the standard datamodule-based call."""
    trainer = mock.Mock()
    trainer.validate.return_value = [{"val/Accuracy": 0.7}, {"val/Accuracy": 0.7}]
    trainer.checkpoint_type = "best"
    model = mock.Mock()
    datamodule = mock.Mock()

    result = functional._evaluate_stage(
        trainer,
        model,
        datamodule,
        stage="validate",
        record_datasets_as_runs=False,
    )

    assert result == [{"val/Accuracy": 0.7}, {"val/Accuracy": 0.7}]
    trainer.validate.assert_called_once_with(
        model=model, datamodule=datamodule, verbose=True, ckpt_path="best"
    )


def test_run_evaluation_session_keeps_dataset_results_grouped_when_recording_disabled() -> None:
    """Tests that full result lists stay grouped when dataset outputs are not recorded as runs."""
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
            record_datasets_as_runs=False,
            verbose=False,
        )

    recorder.update.assert_called_once_with(validation_scores, test_scores)
    recorder.save.assert_called_once_with()
