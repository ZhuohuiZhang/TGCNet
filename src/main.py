import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch
from utils.ex_logging import get_logger
import yaml
from run import run
import sys

SETTINGS['CAPTURE_MODE'] = "no"  # "fd"
logger = get_logger()
ex = Experiment('pymarl-c')
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    config = config_copy(_config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(par, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(par):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del par[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__),
                               "config", subfolder,
                               "{}.yaml".format(config_name)), "r") as F:
            try:
                config_dict_append = yaml.load(F, Loader=yaml.FullLoader)
            except yaml.YAMLError as err:
                assert False, "{}.yaml error: {}".format(config_name, err)
        return config_dict_append


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(par, key, default):
    result = default
    for _i, _v in enumerate(par):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=') + 1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)  # terminal input

    # the number of threads
    torch.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs/
    env_config = _get_config(params, "--env-config", "envs")  # "envs" ./config/envs
    alg_config = _get_config(params, "--alg-config", "algs")  # "algs" ./config/algs
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)
    try:
        map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    except KeyError:
        map_name = parse_command(params, "env_args.key", config_dict['env_args']['key'])
    algo_name = parse_command(params, "name", config_dict['name'])
    file_obs_path = os.path.join(results_path, "sacred", algo_name, map_name)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver(file_obs_path))
    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
