# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

import dllogger
import horovod.tensorflow.keras as hvd
import multiprocessing
import portpicker
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import train_input_fn, eval_input_fn
from data.outbrain.features import PREBATCH_SIZE
from functools import partial
from trainer.utils.arguments import MODE_CPU, MODE_HOROVOD, MODE_PS
from trainer.utils.gpu_affinity import set_affinity


def init_cpu(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(
        full=True,
        args=args,
        logger=logger
    )

    logger.warning('--gpu flag not set, running computation on CPU')


def init_horovod(args, logger):
    hvd.init()

    init_logger(
        full=hvd.rank() == 0,
        args=args,
        logger=logger
    )
    if args.affinity != 'disabled':
        gpu_id = hvd.local_rank()
        affinity = set_affinity(
            gpu_id=gpu_id,
            nproc_per_node=hvd.size(),
            mode=args.affinity
        )
        logger.warning(f'{gpu_id}: thread affinity: {affinity}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)


def init_ps(args, logger):

    init_logger(
        full=True,
        args=args,
        logger=logger
    )

    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)

    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    NUM_WORKERS = 1
    NUM_PS = 1
    cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

    variable_partitioner = (
    tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=NUM_PS))

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)

    return strategy


def init_ps_distributed(args, logger):

    init_logger(
        full=True,
        args=args,
        logger=logger
    )

    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["localhost:19897", "localhost:19898"],
            "ps": ["localhost:19900"],
            "chief": ["localhost:19901"]
        },
        "task": {"type": args.ps_task_type, "index": args.ps_task_index}
    })

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
        os.environ["GRPC_FAIL_FAST"] = "use_caller"

        if cluster_resolver.task_type == "worker":
            physical_devices = tf.config.list_physical_devices('GPU')
            if cluster_resolver.task_id == 0:
                tf.config.set_visible_devices(physical_devices[:4], 'GPU')
            else:
                tf.config.set_visible_devices(physical_devices[4:], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            print("========= List logical devices:")
            print(logical_devices)

        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True)
        server.join()
    
    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)

    variable_partitioner = (
    tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=1))

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)

    return strategy


def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(args.results_dir, args.log_filename)
        os.makedirs(args.results_dir, exist_ok=True)
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                       filename=log_path),
            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
        logger.warning('command line arguments: {}'.format(json.dumps(vars(args))))
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        with open('{}/args.json'.format(args.results_dir), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger.setLevel(logging.ERROR)
        dllogger.init(backends=[])

    dllogger.log(data=vars(args), step='PARAMETER')


def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec, job_name="worker", task_index=i, config=worker_config,
            protocol="grpc")

    for i in range(num_ps):
        tf.distribute.Server(
            cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")
    return cluster_resolver


def create_config(args):
    assert not (args.mode == MODE_CPU and args.amp), \
        'Automatic mixed precision conversion works only with GPU'
    assert not args.benchmark or args.benchmark_warmup_steps < args.benchmark_steps, \
        'Number of benchmark steps must be higher than warmup steps'
    logger = logging.getLogger('tensorflow')

    strategy = None

    if args.mode == MODE_CPU:
        init_cpu(args, logger)
    elif args.mode == MODE_HOROVOD:
        init_horovod(args, logger)
    elif args.in_process:
        strategy = init_ps(args, logger)
    else:
        strategy = init_ps_distributed(args, logger)

    num_gpus = 1 if args.mode != MODE_HOROVOD else hvd.size()
    gpu_id = 0 if args.mode != MODE_HOROVOD else hvd.rank()
    train_batch_size = args.global_batch_size // num_gpus
    eval_batch_size = args.eval_batch_size // num_gpus
    steps_per_epoch = args.training_set_size / args.global_batch_size

    feature_spec = tft.TFTransformOutput(
        args.transformed_metadata_path
    ).transformed_feature_spec()

    train_spec_input_fn = train_input_fn(
        num_gpus=num_gpus,
        id=gpu_id,
        filepath_pattern=args.train_data_pattern,
        feature_spec=feature_spec,
        records_batch_size=train_batch_size // PREBATCH_SIZE,
    )

    def ps_train_spec_input_fn(_):
        return train_input_fn(
            num_gpus=num_gpus,
            id=gpu_id,
            filepath_pattern=args.train_data_pattern,
            feature_spec=feature_spec,
            records_batch_size=train_batch_size // PREBATCH_SIZE,
        )
    
    eval_spec_input_fn = eval_input_fn(
        num_gpus=num_gpus,
        id=gpu_id,
        repeat=None if args.benchmark else 1,
        filepath_pattern=args.eval_data_pattern,
        feature_spec=feature_spec,
        records_batch_size=eval_batch_size // PREBATCH_SIZE
    )

    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train_spec_input_fn,
        'eval_dataset': eval_spec_input_fn,
        'strategy': strategy,
        'ps_train_dataset': ps_train_spec_input_fn,
    }

    return config
