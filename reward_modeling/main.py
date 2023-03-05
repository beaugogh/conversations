from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate


class Example:
    def __init__(self, attr1: str, attr2: float):
        self.attr1 = attr1
        self.attr2 = attr2
        print(attr1, attr2)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    dummy = instantiate(cfg.dummy)
    print()


if __name__ == "__main__":
    app()
