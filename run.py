import os
import sys
import torch
import hydra
from omegaconf import OmegaConf

import stable_ssl

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg):
    """Load the configuration and launch the run."""
    trainer = stable_ssl.instanciate_config(cfg)  # using this instead of hydra.utils.instantiate to create a pickle dump of config (debug hash)
    trainer()
    print(trainer.get_logs())


def entry():
    """Wrapper over main like in the CLI to expand config-path to absolute path before passing to Hydra/OmegaConf."""

    # We need to pass the config path as an absolute path to Hydra.
    if "--config-path" in sys.argv:
        index = sys.argv.index("--config-path")
        if index + 1 < len(sys.argv):
            config_path = sys.argv[index + 1]
            if not os.path.isabs(config_path):
                sys.argv[index + 1] = os.path.abspath(config_path)

    main()


if __name__ == '__main__':
    sys.exit(entry())
