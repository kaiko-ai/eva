"""Example usage:
python cli.py \
    --data_config configs/data/panda.py \
    --model_config configs/model/abmil.py \
    --model_config.input_dim 1024 \
    --download_dir /mnt/localdisk/data/download
"""

from absl import app, flags
from ml_collections import config_flags

config_flags.DEFINE_config_file(
    "data_config", None, "Dataset configuration file.", lock_config=True
)
config_flags.DEFINE_config_file("model_config", None, "Model configuration file.", lock_config=True)
flags.DEFINE_string("download_dir", default=None, help="Work unit directory.")


def main(argv):
    del argv
    data_config = flags.FLAGS.data_config
    model_config = flags.FLAGS.model_config
    download_dir = flags.FLAGS.download_dir

    print(f"DATA CONFIG:\n{data_config}")
    print(f"MODEL CONFIG:\n{model_config}")
    print(f"DOWNLOAD DIR: {download_dir}")


if __name__ == "__main__":
    app.run(main)
