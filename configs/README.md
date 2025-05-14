## Config Modules
Configs are modularized into separate functional components.
1. **data**: Contains all the different datasets, their path, etc.
2. **experiment**: Contains the different pretraining objectives and the corresponding datasets for backbone training and/or eval.
3. **metric**: Metrics for probing the backbone during training and/or eval.
4. **mode**: Different hardware environments.
5. **monitor**: Monitor the embeddings to avoid model collapse, etc.
6. **user**: User specific configurations like logging directory, gpu partition, etc.


## How are configs setup/collated/linked for parsing?
**Step 1**: Hydra loads `default.yaml` as it's specified in the `config_name` of `run.py` hydra main decorator.

**Step 2**: The `defaults` section of the `default.yaml` decides the loading order (top to bottom) of the different config overrides from each of the config modules.

**Step 3**: Within the experiment config module, the `trainer.data` configs are overriden by the data config module because of the `defaults` section.
