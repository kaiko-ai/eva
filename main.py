from eva.core.loggers.json_logger import JSONLogger


metrics_one = {"val/AverageLoss": 0.1}
metrics_two = {"val/AverageLoss": 0.05}
metrics_three = {"val/AverageLoss": 0.03}
metrics_four = {"val/AverageLoss": 0.01}


logger = JSONLogger(root_dir=".", version="")
# logger = JSONLogger(root_dir="az://ml-outputs@kaiko.blob.core.windows.net/experiments/tmp", version="")

logger.log_metrics(metrics=metrics_one)
logger.save()

logger.log_metrics(metrics=metrics_one, step=1)
logger.save()

logger.log_metrics(metrics=metrics_two, step=2)
logger.log_metrics(metrics=metrics_three, step=3)
logger.save()

logger.log_metrics(metrics=metrics_four, step=4)
logger.save()
