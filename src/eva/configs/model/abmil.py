import ml_collections as mlc


class DummyClass:
    def __init__(self, dummy: int):
        pass


def get_config(runlocal=False):
    config = mlc.ConfigDict()

    config.input_dim = 384
    config.hidden_dims = [128, 64]
    config.dummy = DummyClass(1)

    return config
