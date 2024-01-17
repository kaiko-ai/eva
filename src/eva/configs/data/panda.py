import ml_collections as mlc


def get_config(runlocal=False):
    config = mlc.ConfigDict()

    if runlocal:
        config.base_dir = "/home/kaiko/Projects/eva/data/panda"
    else:
        config.base_dir = "az://ml-datasets-public@kaiko.blob.core.windows.net/cpath/wsi/panda"
    config.split_file = "splits/original_split.parquet"

    return config
