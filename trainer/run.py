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

import logging
import os
import time

from tensorflow.python.framework.errors_impl import OutOfRangeError

import dllogger
import horovod.tensorflow as hvd
import multiprocessing
import numpy as np
import portpicker
import tensorflow as tf
from data.outbrain.features import DISPLAY_ID_COLUMN
from tensorflow.python.keras import backend as K
from trainer.utils.arguments import MODE_HOROVOD
from trainer.utils.schedulers import get_schedule


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


# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)


def train(args, model, config):
    logger = logging.getLogger('tensorflow')

    train_dataset = config['train_dataset']
    eval_dataset = config['eval_dataset']
    steps = int(config['steps_per_epoch'])

    schedule = get_schedule(
        args=args,
        steps_per_epoch=steps
    )
    writer = tf.summary.create_file_writer(os.path.join(args.model_dir, 'event_files'))

    deep_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.deep_learning_rate,
        rho=0.5
    )

    wide_optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=args.linear_learning_rate
    )

    compiled_loss = tf.keras.losses.BinaryCrossentropy()
    eval_loss = tf.keras.metrics.Mean()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        model=model,
        current_step=current_step_var
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(args.model_dir, 'checkpoint'),
        max_to_keep=1
    )

    if args.use_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.warning(f'Model restored from checkpoint {args.model_dir}')
            if args.benchmark:
                current_step_var.assign(0)
        else:
            logger.warning(f'Failed to restore model from checkpoint {args.model_dir}')

    if args.amp:
        deep_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            deep_optimizer,
            loss_scale='dynamic'
        )
        wide_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            wide_optimizer,
            loss_scale='dynamic'
        )

    @tf.function
    def train_step(x, y, first_batch):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x, training=True)
            loss = compiled_loss(y, y_pred)
            linear_loss = wide_optimizer.get_scaled_loss(loss) if args.amp else loss
            deep_loss = deep_optimizer.get_scaled_loss(loss) if args.amp else loss

        if args.mode == MODE_HOROVOD:
            tape = hvd.DistributedGradientTape(tape)

        for metric in metrics:
            metric.update_state(y, y_pred)

        linear_vars = model.linear_model.trainable_variables
        dnn_vars = model.dnn_model.trainable_variables
        linear_grads = tape.gradient(linear_loss, linear_vars)
        dnn_grads = tape.gradient(deep_loss, dnn_vars)
        if args.amp:
            linear_grads = wide_optimizer.get_unscaled_gradients(linear_grads)
            dnn_grads = deep_optimizer.get_unscaled_gradients(dnn_grads)

        wide_optimizer.apply_gradients(zip(linear_grads, linear_vars))
        deep_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))
        if first_batch and args.mode == MODE_HOROVOD:
            hvd.broadcast_variables(model.linear_model.variables, root_rank=0)
            hvd.broadcast_variables(model.dnn_model.variables, root_rank=0)
            hvd.broadcast_variables(wide_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(deep_optimizer.variables(), root_rank=0)
        return loss

    @tf.function
    def evaluation_step(x, y):
        predictions = model(x, training=False)
        loss = compiled_loss(y, predictions)

        for metric in metrics:
            metric.update_state(y, predictions)

        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        display_ids = x[DISPLAY_ID_COLUMN]
        display_ids = tf.reshape(display_ids, [-1])
        labels = tf.reshape(y, [-1])
        sorted_ids = tf.argsort(display_ids)
        display_ids = tf.gather(display_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)
        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
        pad_length = 30 - tf.reduce_max(display_ids_ads_count)
        preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
        labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()

        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
        _, predictions_idx = tf.math.top_k(preds_masked, 12)
        indices = tf.math.equal(predictions_idx, labels_masked)
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        display_id_counter.assign_add(shape)
        streaming_map.assign_add(ap_sum)
        return loss

    t0 = None
    t_batch = None

    with writer.as_default():
        for epoch in range(1, args.num_epochs + 1):
            for step, (x, y) in enumerate(train_dataset):
                current_step = np.asscalar(current_step_var.numpy())
                schedule(optimizer=deep_optimizer, current_step=current_step)

                for metric in metrics:
                    metric.reset_states()
                loss = train_step(x, y, epoch == 1 and step == 0)
                if args.mode != MODE_HOROVOD or hvd.rank() == 0:
                    for metric in metrics:
                        tf.summary.scalar(f'{metric.name}', metric.result(), step=current_step)
                    tf.summary.scalar('loss', loss, step=current_step)
                    tf.summary.scalar('schedule', K.get_value(deep_optimizer.lr), step=current_step)
                    writer.flush()

                if args.benchmark:
                    boundary = max(args.benchmark_warmup_steps, 1)
                    if current_step == boundary:
                        t0 = time.time()
                    if current_step > boundary:
                        batch_time = time.time() - t_batch
                        samplesps = args.global_batch_size / batch_time
                        dllogger.log(data={'batch_samplesps': samplesps}, step=(1, current_step))

                        if args.benchmark_steps <= current_step:
                            train_time = time.time() - t0
                            epochs = args.benchmark_steps - max(args.benchmark_warmup_steps, 1)
                            train_throughput = (args.global_batch_size * epochs) / train_time
                            dllogger.log(
                                data={'train_throughput': train_throughput},
                                step=tuple()
                            )
                            return

                else:
                    if current_step % 100 == 0:
                        train_data = {metric.name: f'{metric.result().numpy():.4f}' for metric in metrics}
                        train_data['loss'] = f'{loss.numpy():.4f}'
                        dllogger.log(data=train_data, step=(current_step, args.num_epochs * steps))

                    if step == steps:
                        break

                current_step_var.assign_add(1)
                t_batch = time.time()
            if args.benchmark:
                continue

            for metric in metrics:
                metric.reset_states()
            eval_loss.reset_states()

            for step, (x, y) in enumerate(eval_dataset):
                loss = evaluation_step(x, y)
                eval_loss.update_state(loss)

            map_metric = tf.divide(streaming_map, display_id_counter) if args.mode != MODE_HOROVOD else \
                hvd.allreduce(tf.divide(streaming_map, display_id_counter))

            map_metric = map_metric.numpy()
            eval_loss_reduced = eval_loss.result() if args.mode != MODE_HOROVOD else \
                hvd.allreduce(eval_loss.result())

            metrics_reduced = {
                f'{metric.name}_val': metric.result() if args.mode != MODE_HOROVOD else
                hvd.allreduce(metric.result()) for metric in metrics
            }

            for name, result in metrics_reduced.items():
                tf.summary.scalar(f'{name}', result, step=steps * epoch)
            tf.summary.scalar('loss_val', eval_loss_reduced, step=steps * epoch)
            tf.summary.scalar('map_val', map_metric, step=steps * epoch)
            writer.flush()

            eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
            eval_data.update({
                'loss_val': f'{eval_loss_reduced.numpy():.4f}',
                'streaming_map_val': f'{map_metric:.4f}'
            })
            dllogger.log(data=eval_data, step=(steps * epoch, args.num_epochs * steps))

            if args.mode != MODE_HOROVOD or hvd.rank() == 0:
                manager.save()

            display_id_counter.assign(0)
            streaming_map.assign(0)
        if args.mode != MODE_HOROVOD or hvd.rank() == 0:
            dllogger.log(data=eval_data, step=tuple())


def ps_train(args, model, config):
    logger = logging.getLogger('tensorflow')

    train_dataset = config['ps_train_dataset']
    strategy = config["strategy"]

    writer = tf.summary.create_file_writer(os.path.join(args.model_dir, 'event_files'))

    with strategy.scope():
        deep_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=args.deep_learning_rate,
            rho=0.5
        )

        wide_optimizer = tf.keras.optimizers.Ftrl(
            learning_rate=args.linear_learning_rate
        )

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)

    checkpoint = tf.train.Checkpoint(
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        model=model,
        current_step=current_step_var
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(args.model_dir, 'checkpoint'),
        max_to_keep=1
    )

    if args.use_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.warning(f'Model restored from checkpoint {args.model_dir}')
            if args.benchmark:
                current_step_var.assign(0)
        else:
            logger.warning(f'Failed to restore model from checkpoint {args.model_dir}')

    if args.amp:
        with strategy.scope():
            deep_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                deep_optimizer,
                loss_scale='dynamic'
            )
            wide_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                wide_optimizer,
                loss_scale='dynamic'
            )

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

    @tf.function
    def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(train_dataset)

    per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @tf.function
    def train_step(iterator):
        def replica_fn(x, y):
            with tf.GradientTape(persistent=True) as tape:
                y_pred = model(x, training=True)
                per_example_loss  = tf.keras.losses.BinaryCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
                loss = tf.nn.compute_average_loss(per_example_loss)
                linear_loss = wide_optimizer.get_scaled_loss(loss) if args.amp else loss
                deep_loss = deep_optimizer.get_scaled_loss(loss) if args.amp else loss

                linear_vars = model.linear_model.trainable_variables
                dnn_vars = model.dnn_model.trainable_variables
                linear_grads = tape.gradient(linear_loss, linear_vars)
                dnn_grads = tape.gradient(deep_loss, dnn_vars)
                if args.amp:
                    linear_grads = wide_optimizer.get_unscaled_gradients(linear_grads)
                    dnn_grads = deep_optimizer.get_unscaled_gradients(dnn_grads)

            wide_optimizer.apply_gradients(zip(linear_grads, linear_vars))
            deep_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))
            for metric in metrics:
                metric.update_state(y, y_pred)
            return loss
        
        batch_data, labels = next(iterator)
        losses = strategy.run(replica_fn, args=(batch_data, labels))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    STEPS_PER_EPOCH = 100

    with writer.as_default():
        current_step = np.asscalar(current_step_var.numpy())
        for epoch in range(1, args.num_epochs + 1):
            for metric in metrics:
                metric.reset_states()
            loss = coordinator.schedule(train_step, args=(per_worker_iterator, ))
            print ("Final loss is %f" % loss.fetch())
            for metric in metrics:
                print (f'{metric.name}: {metric.result()}')
            # while True:
            #     try:
            #         for _ in range(STEPS_PER_EPOCH):
            #             loss = coordinator.schedule(train_step, args=(per_worker_iterator, ))
            #         logger.info('***** Before coordinator.join')
            #         coordinator.join()
            #         logger.info('***** After coordinator.join')
            #         current_step += STEPS_PER_EPOCH
            #         for metric in metrics:
            #             tf.summary.scalar(f'{metric.name}', metric.result(), step=current_step)
            #         tf.summary.scalar('loss', loss.fetch(), step=current_step)
            #         tf.summary.scalar('schedule', K.get_value(deep_optimizer.lr), step=current_step)
            #     except OutOfRangeError as e:
            #         logger.info(f'Training epoch {epoch} complete.')
            #         break
        logger.warning('Training all complete.')


def evaluate(args, model, config):
    logger = logging.getLogger('tensorflow')

    deep_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.deep_learning_rate,
        rho=0.5
    )

    wide_optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=args.linear_learning_rate
    )

    compiled_loss = tf.keras.losses.BinaryCrossentropy()
    eval_loss = tf.keras.metrics.Mean()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    if args.amp:
        deep_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            deep_optimizer,
            loss_scale='dynamic'
        )
        wide_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            wide_optimizer,
            loss_scale='dynamic'
        )

    current_step = 0
    current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        model=model,
        current_step=current_step_var
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(args.model_dir, 'checkpoint'),
        max_to_keep=1
    )

    if args.use_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            logger.warning(f'Model restored from checkpoint {args.model_dir}')
        else:
            logger.warning(f'Failed to restore model from checkpoint {args.model_dir}')

    @tf.function
    def evaluation_step(x, y):
        predictions = model(x, training=False)
        loss = compiled_loss(y, predictions)

        for metric in metrics:
            metric.update_state(y, predictions)

        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        display_ids = x[DISPLAY_ID_COLUMN]
        display_ids = tf.reshape(display_ids, [-1])
        labels = tf.reshape(y, [-1])
        sorted_ids = tf.argsort(display_ids)
        display_ids = tf.gather(display_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)
        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
        pad_length = 30 - tf.reduce_max(display_ids_ads_count)
        preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
        labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()

        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
        _, predictions_idx = tf.math.top_k(preds_masked, 12)
        indices = tf.math.equal(predictions_idx, labels_masked)
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        display_id_counter.assign_add(shape)
        streaming_map.assign_add(ap_sum)
        return loss

    eval_dataset = config['eval_dataset']

    t0 = None
    t_batch = None

    for step, (x, y) in enumerate(eval_dataset):
        loss = evaluation_step(x, y)
        eval_loss.update_state(loss)
        if args.benchmark:
            boundary = max(args.benchmark_warmup_steps, 1)
            if current_step == boundary:
                t0 = time.time()
            if current_step > boundary:
                batch_time = time.time() - t_batch
                samplesps = args.eval_batch_size / batch_time
                if args.mode != MODE_HOROVOD or hvd.rank() == 0:
                    dllogger.log(data={'batch_samplesps': samplesps}, step=(1, current_step))

                if args.benchmark_steps <= current_step:
                    valid_time = time.time() - t0
                    epochs = args.benchmark_steps - max(args.benchmark_warmup_steps, 1)
                    valid_throughput = (args.eval_batch_size * epochs) / valid_time
                    if args.mode != MODE_HOROVOD or hvd.rank() == 0:
                        dllogger.log(
                            data={'validation_throughput': valid_throughput},
                            step=tuple()
                        )
                    return

        else:
            if step % 100 == 0:
                valid_data = {metric.name: f'{metric.result().numpy():.4f}' for metric in metrics}
                valid_data['loss'] = f'{loss.numpy():.4f}'
                if args.mode != MODE_HOROVOD or hvd.rank() == 0:
                    dllogger.log(data=valid_data, step=(step,))
        current_step += 1
        t_batch = time.time()

    map_metric = tf.divide(streaming_map, display_id_counter) if args.mode != MODE_HOROVOD else \
        hvd.allreduce(tf.divide(streaming_map, display_id_counter))
    eval_loss_reduced = eval_loss.result() if args.mode != MODE_HOROVOD else \
        hvd.allreduce(eval_loss.result())

    metrics_reduced = {
        f'{metric.name}_val': metric.result() if args.mode != MODE_HOROVOD else
        hvd.allreduce(metric.result()) for metric in metrics
    }

    eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
    eval_data.update({
        'loss_val': f'{eval_loss_reduced.numpy():.4f}',
        'streaming_map_val': f'{map_metric.numpy():.4f}'
    })

    dllogger.log(data=eval_data, step=(step,))
