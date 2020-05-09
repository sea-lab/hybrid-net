import tensorflow as tf


@tf.function
def find_change_indices(y):
    # y = tf.reshape(y, [-1])
    if y.get_shape()[0] == 0:
        return tf.constant([], dtype='int32')

    shifted = tf.concat([y[:1], y[:-1]], 0)
    changes = tf.where(tf.reshape(shifted != y, [-1]))

    return tf.cast(tf.reshape(changes, [-1]), 'int32')


@tf.function
def remove_clutter_one_sample(prediction, t=5):
    get_shape_result = prediction.get_shape()
    tf_shape_result = tf.shape(prediction)

    if get_shape_result is not None and len(get_shape_result) > 0:
        L = int(get_shape_result[0])
    else:
        L = int(tf_shape_result[0])

    changes = tf.concat([[0], find_change_indices(prediction), [L - 1]], axis=0)

    if tf.shape(changes)[0] > 2:
        changes = changes[tf.concat((changes[1:] - changes[:-1] > t, [False]), axis=0)]

    changes, _ = tf.unique(changes)
    repeats = tf.concat([changes[1:] - changes[:-1], [L - changes[-1]]], axis=0)
    adjust_for_first_item = tf.concat(([changes[0]], tf.zeros(tf.shape(repeats) - 1, dtype='int32')), axis=0)
    repeats += adjust_for_first_item

    new_prediction = tf.repeat(
        tf.gather(prediction, changes),
        repeats
    )

    tf.assert_equal(tf.reduce_sum(repeats), L)
    new_prediction.set_shape([L])
    return new_prediction

#
# @tf.function
# def precision_like_loss(y_true, y_pred):
#     y_true = tf.math.argmax(y_true, axis=-1)  # batch x L
#     y_pred = tf.math.argmax(y_pred, axis=-1)  # batch x L
#     stacked = tf.stack([y_pred, y_true], axis=1)  # batch x 2 x L
#     results = tf.map_fn(positives_sum, stacked, dtype='int32')  # batch x 2
#     tp = results[:, 0]  # batch
#     fp = results[:, 1]  # batch
#
#     all_losses = tf.math.divide_no_nan(tf.cast(tp, 'float32'), tf.cast(tp + fp, 'float32'))  # batch
#     return tf.math.reduce_mean(all_losses, axis=0)  # scalar
#
#
# @tf.function
# def sum_losses(y_true, y_pred):
#     return precision_like_loss(y_true, y_pred) + soft_dice_loss(y_true, y_pred)


# @tf.function
# def remove_clutters(input, max_length):
#     _, n_samples, n_classes = input.get_shape().as_list()
#     predictions = tf.math.argmax(input, axis=-1)
#
#     @tf.function
#     def remove_clutter_one_sample(prediction):
#         changes = tf.concat([[0], find_change_indices(prediction), [max_length]], axis=0)
#
#         if tf.shape(changes)[0] > 2:
#             changes = changes[changes[1:] - changes[:-1] > 5]
#
#             if changes[0] != 0:
#                 changes = tf.concat([[0], changes], axis=0)
#
#         new_prediction = tf.repeat(
#             tf.gather(prediction, changes),
#             tf.concat([changes[1:] - changes[:-1], [max_length - changes[-1]]], axis=0)
#         )
#         new_prediction.set_shape([max_length])
#
#         return new_prediction
#
#     changes = tf.map_fn(remove_clutter_one_sample, predictions)
#     return changes

#
# def clear_clutters(changes, t=5):
#     if tf.shape(changes)[0] > 2:
#         changes = changes[changes[1:] - changes[:-1] > t]
#     return changes


def soft_dice_loss(y_true, y_pred):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    denominator = tf.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)

    return 1 - tf.math.reduce_mean(tf.math.divide_no_nan(numerator, denominator))  # average over classes and batch

# @tf.function
# def positives_alpha(Ys, tolerance=1, alpha=0.5):
#     y_pred = Ys[0]
#     y_true = Ys[1]
#
#     true_positive = 0.0
#     false_positive = 0.0
#
#     changes_true = find_change_indices(y_true)
#     changes_pred = find_change_indices(y_pred)
#     clear_clutters(changes_pred)
#
#     if tf.shape(changes_true)[0] == 0:
#         if tf.shape(changes_pred)[0] == 0:
#             true_positive = 1.
#         else:
#             false_positive = tf.cast(tf.shape(changes_pred)[0], 'float32')
#     else:
#         for change in changes_pred:
#             # Can be made more efficient, binary tree?
#             closest_index = tf.argmin(tf.abs(changes_true - change))
#             closest_value = tf.gather(changes_true, closest_index)
#             if tf.abs(closest_value - change) <= tolerance:
#                 new_state_true = tf.gather(y_true, closest_value)
#                 new_state_pred = tf.gather(y_pred, change)
#
#                 if new_state_true == new_state_pred:
#                     true_positive += 1.0
#                 else:
#                     true_positive += alpha
#             else:
#                 false_positive += 1.0
#
#     return tf.convert_to_tensor([true_positive, false_positive], dtype='float32')

#
# @tf.function
# def positives_sum(Ys):
#     y_pred = Ys[0]
#     y_true = Ys[1]
#
#     changes_true = find_change_indices(y_true)
#     # changes_pred = find_change_indices(y_pred)
#     # # clear_clutters(changes_pred)
#     changes_pred = find_change_indices(remove_clutter_one_sample(y_pred, tf.shape(y_pred)[0]))
#
#     if tf.shape(changes_true)[0] == 0:
#         true_positive = 0
#         if tf.shape(changes_pred)[0] == 0:
#             false_positive = 0
#         else:
#             false_positive = tf.reduce_sum(changes_pred)
#     else:
#         changes_pred = tf.reshape(changes_pred, [-1, 1])
#         changes_true = tf.reshape(changes_true, [1, -1])
#
#         distances = tf.abs(changes_pred - changes_true)  # axis0 = pred, axis1 = true
#
#         false_positive = tf.reduce_sum(tf.reduce_min(distances, axis=1))
#         true_positive = tf.reduce_sum(tf.reduce_min(distances, axis=0))
#
#     return tf.convert_to_tensor([true_positive, false_positive])


def create_positives(tolerance=25):
    @tf.function
    def positives(Ys):
        y_pred = Ys[0]
        y_true = Ys[1]

        changes_true = find_change_indices(y_true)
        changes_pred = find_change_indices(remove_clutter_one_sample(y_pred))

        if tf.shape(changes_true)[0] == 0 and tf.shape(changes_pred)[0] == 0:
            true_positive = 1
            false_positive = 0
        else:
            changes_pred = tf.reshape(changes_pred, [-1, 1])
            changes_true = tf.reshape(changes_true, [1, -1])

            distances = tf.abs(changes_pred - changes_true)  # axis0 = pred, axis1 = true
            false_positive = tf.reduce_sum(tf.cast(tf.reduce_min(distances, axis=1) > tolerance, 'int32'))
            true_positive = tf.reduce_sum(tf.cast(tf.reduce_min(distances, axis=0) <= tolerance, 'int32'))

        return tf.stack((true_positive, false_positive))

    return positives


def create_negatives(tolerance=25):
    @tf.function
    def negatives(Ys):
        y_pred = Ys[0]
        y_true = Ys[1]

        changes_true = find_change_indices(y_true)
        # changes_pred = find_change_indices(y_pred)
        # clear_clutters(changes_pred)
        changes_pred = find_change_indices(remove_clutter_one_sample(y_pred))

        if tf.shape(changes_pred)[0] == 0:
            return tf.cast(tf.shape(changes_true)[0], 'int32')
        else:
            changes_pred = tf.reshape(changes_pred, [-1, 1])
            changes_true = tf.reshape(changes_true, [1, -1])
            min_distances = tf.reduce_min(tf.abs(changes_pred - changes_true), axis=0)

            return tf.reduce_sum(tf.cast(min_distances > tolerance, 'int32'))

    return negatives


class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='precision_with_tolerance', dtype=None, tolerance=1):# , alpha=.5):
        super(Precision, self).__init__(name, dtype)
        self.tolerance = tolerance
        # self.alpha = alpha
        self.sum_scores = self.add_weight(name=f'{name}_sum_scores', initializer='zeros', dtype='float32')
        self.count = self.add_weight(name=f'{name}_count', initializer='zeros', dtype='int32')
        self.positives = create_positives(tolerance)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred.shape.assert_is_compatible_with(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        stacked = tf.stack([y_pred, y_true], axis=1)  # batch x 2 x L
        results = tf.map_fn(self.positives, stacked, dtype='int32')  # batch x 2
        tp, fp = results[:, 0], results[:, 1]

        score = tf.math.divide_no_nan(tf.cast(tp, 'float32'), tf.cast(tp + fp, 'float32'))  # shape = batch
        self.sum_scores.assign_add(tf.reduce_sum(score))
        self.count.assign_add(tf.shape(score)[0])
 
    def result(self):
        result = tf.math.divide_no_nan(self.sum_scores, tf.cast(self.count, 'float32'))
        return result


class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='recall_with_tolerance', dtype=None, tolerance=1):
        super(Recall, self).__init__(name, dtype)
        self.tolerance = tolerance
        self.sum_scores = self.add_weight(name=f'{name}_sum_scores', initializer='zeros', dtype='float32')
        self.count = self.add_weight(name=f'{name}_recall_count', initializer='zeros', dtype='int32')
        self.positives = create_positives(tolerance)
        self.negatives = create_negatives(tolerance)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        if not y_pred.shape.is_compatible_with(y_true.shape):
            tf.print('p',tf.shape(y_pred), end=' ')
            tf.print('t',tf.shape(y_true), end=' ')
            raise ValueError("Shapes %s and %s are incompatible" % (y_pred.shape, y_true.shape))
        
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        stacked = tf.stack([y_pred, y_true], axis=1)  # batch x 2 x L
        tp = tf.map_fn(self.positives, stacked, dtype='int32')[:, 0]  # batch x 2
        fn = tf.map_fn(self.negatives, stacked, dtype='int32')  # batch x 2

        score = tf.math.divide_no_nan(tf.cast(tp, 'float32'), tf.cast(tp + fn, 'float32'))

        self.sum_scores.assign_add(tf.reduce_sum(score))
        self.count.assign_add(tf.shape(score)[0])

    def result(self):
        # result = tf.math.divide_no_nan(self.true_positives,
        #                                self.true_positives + tf.cast(self.false_negatives, 'float32'))
        result = tf.math.divide_no_nan(self.sum_scores, tf.cast(self.count, 'float32'))
        return result


def F1(prec, recl):
    @tf.function
    def f1_score(y_true, y_pred):
        precision = prec.result()
        recall = recl.result()
        if precision is not None and recall is not None:
            return tf.math.divide_no_nan(2 * precision * recall, (precision + recall))
        else:
            return 0

    return f1_score


class ClassPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='class_precision', dtype=None):
        super(ClassPrecision, self).__init__(name, dtype)
        self.sum_scores = self.add_weight(name=f'{name}_sum_scores', initializer='zeros', dtype='float32')
        self.count = self.add_weight(name=f'{name}_count', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size, _, classes = y_pred.get_shape()
        if batch_size is None:
            shp = tf.shape(y_pred)
            batch_size, classes = shp[0], shp[1]
        y_true = tf.math.argmax(y_true, axis=-1)  # batch x L
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred.shape.assert_is_compatible_with(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        scores = []
        for C in range(classes):
            C = tf.cast(C, 'int64')
            declaredC = tf.math.count_nonzero(tf.equal(y_pred, C), axis=1)
            correctlyC = tf.math.count_nonzero(tf.logical_and(tf.equal(y_pred, C), tf.equal(y_true, C)), axis=1)

            score = tf.math.divide_no_nan(tf.cast(correctlyC, 'float32'), tf.cast(declaredC, 'float32'))
            score = score + tf.cast(tf.equal(declaredC, 0), score.dtype)  # 0 / 0 = 1
            scores.append(score)

        scores = tf.reduce_mean(tf.stack(scores, axis=1), axis=1)

        self.sum_scores.assign_add(tf.reduce_sum(scores))
        self.count.assign_add(batch_size)

    def result(self):
        result = tf.math.divide_no_nan(self.sum_scores, tf.cast(self.count, 'float32'))
        return result


class ClassRecall(tf.keras.metrics.Metric):
    def __init__(self, name='class_recall', dtype=None):
        super(ClassRecall, self).__init__(name, dtype)
        self.sum_scores = self.add_weight(name=f'{name}_sum_scores', initializer='zeros', dtype='float32')
        self.count = self.add_weight(name=f'{name}_count', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size, _, classes = y_pred.get_shape()
        if batch_size is None:
            shp = tf.shape(y_pred)
            batch_size, classes = shp[0], shp[1]
        y_true = tf.math.argmax(y_true, axis=-1)  # batch x L
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred.shape.assert_is_compatible_with(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        scores = []
        for C in range(classes):
            trueC = tf.math.count_nonzero(tf.equal(y_true, C), axis=1)
            correctlyC = tf.math.count_nonzero(tf.logical_and(tf.equal(y_pred, C), tf.equal(y_true, C)), axis=1)

            score = tf.math.divide_no_nan(tf.cast(correctlyC, 'float32'), tf.cast(trueC, 'float32'))
            score = score + tf.cast(tf.equal(trueC, 0), score.dtype)  # 0 / 0 = 1
            scores.append(score)

        scores = tf.reduce_mean(tf.stack(scores, axis=1), axis=1)

        self.sum_scores.assign_add(tf.reduce_sum(scores))
        self.count.assign_add(batch_size)

    def result(self):
        result = tf.math.divide_no_nan(self.sum_scores, tf.cast(self.count, 'float32'))
        return result
