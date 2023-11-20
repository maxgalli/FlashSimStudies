from omegaconf import OmegaConf
from comet_ml import Experiment


def flatten_dict(d, parent_key='', sep='.'):
   items = []
   for k, v in d.items():
       new_key = parent_key + sep + k if parent_key else k
       if isinstance(v, dict):
           items.extend(flatten_dict(v, new_key, sep=sep).items())
       else:
           items.append((new_key, v))
   return dict(items)


def setup_comet_logger(name, cfg):
    comet_logger = Experiment(
        api_key="DzzVXiirHMuZBc2iIketfZWbm",
        workspace="maxgalli",
        project_name="flashsimstudies",
        #experiment_name="",
        #save_dir="",
    )
    comet_logger.set_name(name)
    # rearrange the dict such that if a key is a dict, it is flattened
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = flatten_dict(cfg)
    for k, v in cfg.items():
        comet_logger.log_parameter(k, v)
    return comet_logger

