"""Train few shot model multiple times with different seeds.

The initial seed will be used to generate seeds for multiple experiments. The
behavior of multiple train is similar to repeat normal training pipeline
multiple times, except the output files from multiple experiments are saved in
same parent directory.
"""
import argparse
import copy
import json
import os
import os.path as osp
import time
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env, get_root_logger

import mmfewshot  # noqa: F401, F403
from mmfewshot import __version__
from mmfewshot.detection.apis import set_random_seed, train_detector
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train FewShot model in multiple '
        'times with different seeds.')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('times', type=int, help='repeat times of experiments')
    parser.add_argument(
        '--start',
        default=0,
        type=int,
        help='number of resume experiment times')
    parser.add_argument(
        '--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    base_cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if base_cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**base_cfg['custom_imports'])
    # set cudnn_benchmark
    if base_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.resume_from is not None:
        base_cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        base_cfg.gpu_ids = args.gpu_ids
    else:
        base_cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # init distributed env first, since logger depends on the dist info.
    rank, world_size = get_dist_info()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **base_cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        base_cfg.gpu_ids = range(world_size)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        base_cfg.work_dir = args.work_dir
    elif base_cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        base_cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0])

    # to make all gpu load same dataset under same random seed
    # all random operation on dataset loading use numpy random
    if args.seed is not None:
        seed = args.seed
    elif base_cfg.seed is not None:
        seed = base_cfg.seed
    else:
        seed = 42
        Warning(f'When using DistributedDataParallel, each rank will '
                f'initialize different random seed. It will cause different'
                f'random action for each rank. In few shot setting, novel '
                f'shots may be generated by random sampling. If all rank do '
                f'not use same seed, each rank will sample different data.'
                f'It will cause UNFAIR data usage. Therefore, seed is set '
                f'to {seed} for default.')

    np.random.seed(int(seed))
    all_seeds = np.random.randint(0, 1000000, args.times).tolist()
    print(f'using seeds for {args.times} times training: ', all_seeds)

    # train with saved dataset
    resume_ann_cfg = None
    if base_cfg.get('multi_train', None) is not None:
        resume_ann_cfg = base_cfg.multi_train.get('resume_ann_cfg', None)
        if resume_ann_cfg is not None:
            assert isinstance(resume_ann_cfg, dict) and \
                   len(resume_ann_cfg.keys()) == args.times, \
                   'multiple train resume ann_cfg mismatch repeat times.'

    for i in range(args.start, int(args.times)):
        cfg = copy.deepcopy(base_cfg)
        # set experiment under different subdirectory
        cfg.work_dir = osp.join(base_cfg.work_dir, f'times_{i}')
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        base_cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = base_cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{base_cfg.pretty_text}')
        # set random seed
        logger.info(f'Set random seed to {all_seeds[i]}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(all_seeds[i], deterministic=args.deterministic)
        cfg.seed = all_seeds[i]
        meta['meta_seed'] = seed
        meta['seed'] = all_seeds[i]
        meta['exp_name'] = osp.basename(args.config + f'_times_{i}')

        # build_detector will do three things, including building model,
        # initializing weights and freezing parameters (optional).
        model = build_detector(cfg.model, logger=logger)

        # if dataset is wrapped by more than one wrapper, please
        # modify follow code mutually.
        if resume_ann_cfg is not None:
            if cfg.data.train.get('ann_cfg', None) is None:
                cfg.data.train.dataset.ann_cfg = resume_ann_cfg[i]
            else:
                cfg.data.train.ann_cfg = resume_ann_cfg[i]
        # build_dataset will do two things, including building dataset
        # and saving dataset into json file (optional).
        datasets = [
            build_dataset(cfg.data.train, rank=rank, timestamp=timestamp)
        ]

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmfewshot version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmfewshot_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

    # get multiple evaluation
    log_file = osp.join(base_cfg.work_dir, 'eval.log')
    logger = get_root_logger(log_file=log_file, log_level=base_cfg.log_level)
    eval_result_list = []
    for i in range(int(args.times)):
        work_dir = osp.join(base_cfg.work_dir, f'times_{i}')
        json_log = sorted(
            [file for file in os.listdir(work_dir) if 'log.json' in file])[-1]
        json_log = os.path.join(work_dir, json_log)
        eval_result = None
        with open(json_log, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                for k in line.keys():
                    if 'AP' in k:
                        eval_result = line
        if eval_result is None:
            logger.info(f'missing times {i} experiment eval result.')
        else:
            for k in [
                    'mode', 'epoch', 'iter', 'lr', 'memory', 'time',
                    'data_time'
            ]:
                if k in eval_result.keys():
                    eval_result.pop(k)
            logger.info(' '.join(
                [f'experiment {i} last eval result:'] +
                [f'{k}: {eval_result[k]}' for k in eval_result.keys()]))
            eval_result_list.append(eval_result)
    num_result = len(eval_result_list)
    if num_result == 0:
        logger.info('found zero eval result.')
        return
    for k in eval_result_list[0].keys():
        eval_result_list[0][k] = sum(
            [eval_result_list[i][k] for i in range(num_result)]) / num_result
    logger.info(f'{num_result} times avg eval result:')
    logger.info(' '.join([f'average {num_result} eval results:'] + [
        f'Avg {k}: {eval_result_list[0][k]}'
        for k in eval_result_list[0].keys()
    ]))


if __name__ == '__main__':
    main()