from typing import Dict, Any, Tuple, Union, Callable, Iterable
import itertools

import pandas as pd
import tensorflow as tf

import metrics
import model_helper

metrics_reporting = [
    metrics.Precision(name='prec_15', tolerance=5),
    metrics.Recall(name='recall_15', tolerance=5),
    metrics.Precision(name='prec_15', tolerance=15),
    metrics.Recall(name='recall_15', tolerance=15),
    metrics.Precision(name='prec_25', tolerance=25),
    metrics.Recall(name='recall_25', tolerance=25),
    metrics.ClassPrecision(),
    metrics.ClassRecall(),
]


def evaluate_model(_model, dataset, metrics_=None):  # TODO: probably belongs to somewhere else
    if not metrics_:
        metrics_ = metrics_reporting

    batch_results_mean = pd.Series(dtype='float16')
    n = 0
    for bi, data in dataset.enumerate():
        ins, ground_truth = data
        prediction = _model.predict_on_batch(ins)
        ground_truth = tf.squeeze(ground_truth)

        metric_names = list()
        metric_results = list()
        for metric in metrics_:
            metric.reset_states()
            metric.update_state(ground_truth, prediction)

            metric_names.append(metric.name)
            metric_results.append(metric.result())

        batch_results = pd.Series(metric_results, index=metric_names, dtype='float16')
        batch_results_mean = batch_results_mean.add(batch_results, fill_value=0.)
        n += 1
    batch_results_mean /= n

    return batch_results_mean


def trainable_count(model):
    weights = model.trainable_weights
    weight_ids = set()
    total = 0
    for w in weights:
        if id(w) not in weight_ids:
            weight_ids.add(id(w))
            total += int(tf.keras.backend.count_params(w))
    return total


def freeze_last_n_layers(model: tf.keras.Model, n: int):
    print('unfreezing', n, 'layers')
    N = len(model.layers)
    trainable = [False] * (N - n) + [True] * n
    for layer, t in zip(model.layers, trainable):
        layer: tf.keras.layers.Layer
        layer.trainable = t

    for layer, t in zip(model.layers, trainable):
        assert layer.trainable == t


def prepare_transfer_model(original_model: tf.keras.Model, original_model_params: Dict[str, Any], layers_to_drop:int =0, **new_params) -> tf.keras.Model:
    if layers_to_drop == 0:
        return original_model

    new_model_params = original_model_params.copy()
    new_model_params.update(new_params)

    new_model = model_helper.make_model(**new_model_params)

    for old_layer, new_layer in zip(original_model.layers[:-layers_to_drop], new_model.layers):
        new_layer: tf.keras.layers.Layer
        old_layer: tf.keras.layers.Layer

        new_layer.set_weights(old_layer.get_weights())

    freeze_last_n_layers(new_model, layers_to_drop)

    return new_model


def train_and_test_transfer_model(X: Union[int, float], optimizer: Union[str, tf.keras.optimizers.Optimizer],
                                  training_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                                  loss: Union[str, tf.keras.losses.Loss, Callable],
                                  metrics_: Iterable[Union[str, tf.keras.metrics.Metric]],
                                  plot: bool=False, summary: bool=False, epochs=100, unfreeze_schedule=None,
                                  **model_args) -> Tuple[tf.keras.Model, pd.Series, pd.DataFrame]:
    if isinstance(X, int):
        #ke hichi
        pass
    elif isinstance(X, float):
        raise NotImplemented("Not yet implemented fractions!")
    else:
        raise ValueError(f"Expected int or float, got {type(X)}")

    new_model = prepare_transfer_model(**model_args)
    new_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_)

    if plot:
        tf.keras.utils.plot_model(new_model, show_shapes=True)
    if summary:
        new_model.summary()

    training_subset = training_dataset.unbatch().take(X).batch(25)
    _history = new_model.fit(training_subset,
                             epochs=epochs,
                             validation_data=validation_dataset,
                             callbacks=[
                                UnfreezeLayersSchedulerCallback(unfreeze_schedule),
                                # tensorboard_callback,
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                             ])

    return (new_model,
           evaluate_model(new_model, validation_dataset),
           pd.DataFrame(_history.history))


class UnfreezeLayersSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, schedule, *args, **kwargs):
        super(UnfreezeLayersSchedulerCallback, self).__init__(*args, **kwargs)
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not self.schedule:
            return
        self.model: tf.keras.Model
        model = self.model

        n = self.schedule.get(epoch, None)
        if n is None:
            return

        params_before = trainable_count(model)
        freeze_last_n_layers(model, n)
        model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
        assert params_before != trainable_count(model)


def alternate_training_loop(
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    training_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
    new_model: tf.keras.Model,
    loss: Union[str, tf.keras.losses.Loss, Callable],
    metrics_: Iterable[Union[str, tf.keras.metrics.Metric]],
    epochs=100,
    transfer_learning_plan=Dict[int, Tuple[float, int]]
):
    history = tf.keras.callbacks.History()
    history.set_model(new_model)
    history.on_train_begin()
    for epoch in range(epochs):
        history.on_epoch_begin(epoch)
        if epoch in transfer_learning_plan:
            new_lr, new_layers = transfer_learning_plan[epoch]
            optimizer = new_model.optimizer
            if new_lr is not None:
                # Change learning rate
                optimizer = tf.keras.optimizers.Adam(learning_rate=new_lr)
            if new_layers is not None:
                freeze_last_n_layers(new_model, new_layers)

            new_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_)

        for bi, (inputs, targets) in enumerate(training_dataset):
            history.on_batch_begin(bi)
            # Open a GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass.
                predictions = new_model(inputs)
                # Compute the loss value for this batch.
                loss_value = new_model.loss(targets, predictions)

            # Get gradients of loss wrt the *trainable* weights.
            gradients = tape.gradient(loss_value, new_model.trainable_weights)
            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, new_model.trainable_weights))
            history.on_batch_end(bi)

        evaluation_results = evaluate_model(new_model, validation_dataset, new_model.metrics)
        print(f'epoch = {epoch}, evaluation = {evaluation_results}')
        history.on_epoch_end(epoch, evaluation_results.to_dict())

    return history

