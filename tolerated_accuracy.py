import tensorflow as tf

find_change_indices = lambda y: tf.where(tf.concat([y[:1], y[:-1]], 0) != y)


@tf.function
def positives(Ys, tolerance=1, alpha=0.5):
    y_pred = Ys[0]
    y_true = Ys[1]

    true_positive = 0.0
    false_positive = 0.0

    changes_true = find_change_indices(y_true)
    changes_pred = find_change_indices(y_pred)

    for change in changes_pred:
        # Can be made more efficient, binary tree?
        closest_index = tf.argmin(tf.abs(changes_true - change))

        closest_value = tf.gather(changes_true, closest_index)

        if tf.abs(closest_value - change) <= tolerance:
            new_state_true = tf.gather(y_true, closest_value)
            new_state_pred = tf.gather(y_pred, change)

            if new_state_true == new_state_pred:
                true_positive += 1.0
            else:
                true_positive += alpha
        else:
            false_positive += 1.0

    return tf.convert_to_tensor([true_positive, false_positive], dtype='float32')


class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='precision_with_tolerance', dtype=None, tolerance=1, alpha=0.5):
        super(Precision, self).__init__(name, dtype)
        self.tolerance = tolerance
        self.alpha = alpha
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype='float32')
        self.false_positives = self.add_weight(name='fp', initializer='zeros', dtype='float32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred.shape.assert_is_compatible_with(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        stacked = tf.stack([y_pred, y_true], axis=1)
        results = tf.map_fn(positives, stacked, dtype='float32')
        self.true_positives.assign_add(tf.reduce_sum(results[:, 0]))
        self.false_positives.assign_add(tf.reduce_sum(results[:, 1]))
 
    def result(self):
        result = tf.math.divide_no_nan(self.true_positives,
                                       self.true_positives + self.false_positives)
        return result


@tf.function
def negatives(Ys, tolerance=1):
    y_pred = Ys[0]
    y_true = Ys[1]

    changes_true = find_change_indices(y_true)
    changes_pred = find_change_indices(y_pred)

    counter = 0
    for change in changes_true:
        if tf.reduce_min(tf.abs(changes_pred - change)) > tolerance:
            counter += 1

    return counter

class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='recall_with_tolerance', dtype=None, tolerance=1):
        super(Recall, self).__init__(name, dtype)
        self.tolerance = tolerance
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype='float32')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_pred.shape.assert_is_compatible_with(y_true.shape)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        stacked = tf.stack([y_pred, y_true], axis=1)
        results = tf.map_fn(positives, stacked, dtype='float32')
        self.true_positives.assign_add(tf.reduce_sum(results[:, 0]))
        results = tf.map_fn(negatives, stacked, dtype='int32')
        self.false_negatives.assign_add(tf.reduce_sum(results))
 
    def result(self):
        result = tf.math.divide_no_nan(self.true_positives,
                                       self.true_positives + tf.cast(self.false_negatives, 'float32'))
        return result


def F1(prec, recl):
    @tf.function
    def f1_score(y_true, y_pred):
        precision = prec.result()
        recall = recl.result()
        return tf.math.divide_no_nan(2 * precision * recall, (precision + recall))

    return f1_score


prec = Precision(tolerance=5, alpha=.5)
recl = Recall(tolerance=5)

metrics = [
    'accuracy',
    prec,
    recl,
    F1(prec, recl)
]