from thermal_flow.training.tracking import WandbLogger


def test_disabled_logger_is_a_no_op() -> None:
    logger = WandbLogger(enabled=False)
    logger.log({"loss": 1.0}, step=0)
    logger.finish()


def test_enabled_logger_uses_wandb_disabled_mode_without_network() -> None:
    with WandbLogger(enabled=True, project="thermal-flow-tests", mode="disabled") as logger:
        logger.log({"loss": 1.0}, step=0)


def test_context_manager_calls_finish_on_exit() -> None:
    with WandbLogger(enabled=False) as logger:
        assert logger.enabled is False
