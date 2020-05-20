import pprint
import logging
import theano
import theano.tensor as T
import lasagne


def masked_loss_with_regularization(network, target_var, known_mask, objective,
                                    regularization=None, deterministic=False):
    """
    Custom loss used with ThresholdStochasticNegativeError
    Args:
        network: lasagne network
        target_var: theano variable for targets (known values/outputs/targets variable)
        known_mask: boolean matrix of known values
        objective: lasagne.objectives function or any function that takes a prediction and known value and returns loss
        regularization: dict like {'kwargs': {'lambda', 0.000001}, 'penalty': lasagne.regularization.l2}
        deterministic: true if test or validation, false if training (default)

    Returns:
        normalized sum of loss of prediction and truth
    """
    prediction = lasagne.layers.get_output(network, deterministic=deterministic)
    loss, lmask = objective(prediction, target_var)
    if regularization:
        lmbda = regularization['lambda']
        penalty = regularization['penalty']
        kwargs = {} if 'kwargs' not in regularization else regularization['kwargs']
        loss += lmbda * lasagne.regularization.regularize_network_params(network, penalty, **kwargs)
    return lasagne.objectives.aggregate(loss, weights=lmask | known_mask, mode='normalized_sum')


def generic_loss_with_regularization(network, target_var, known_mask, objective,
                                       regularization=None, deterministic=False):
    """

    Args:
        network: lasagne network
        target_var: theano variable for targets (known values/outputs/targets variable)
        known_mask: boolean matrix of known values
        objective: lasagne.objectives function or any function that takes a prediction and known value and returns loss
        regularization: dict like {'kwargs': {'lambda', 0.000001}, 'penalty': lasagne.regularization.l2}
        deterministic: true if test or validation, false if training (default)

    Returns:
        normalized sum of loss of prediction and truth
    """
    prediction = lasagne.layers.get_output(network, deterministic=deterministic)
    loss = objective(prediction, target_var)
    if regularization:
        lmbda = regularization['lambda']
        penalty = regularization['penalty']
        kwargs = {} if 'kwargs' not in regularization else regularization['kwargs']
        loss += lmbda * lasagne.regularization.regularize_network_params(network, penalty, **kwargs)
    return lasagne.objectives.aggregate(loss, weights=known_mask, mode='normalized_sum')


class GenericUpdateWithRegularization(object):
    def __init__(self, objective, update, regularization, loss_function=generic_loss_with_regularization):
        """
        Initialize generic train and test generator with objective, update and regularization
        Args:
            objective: a lasagne.objectives or any function that takes a prediction and known value and returns loss
            update: dict like {'kwargs': {'learning_rate', 0.1}, 'function': lasagne.regularization.l2}
            regularization: dict like {'kwargs': {'lambda', 0.000001}, 'penalty': lasagne.regularization.l2}
        """
        self.objective = objective
        self.update = update
        self.regularization = regularization
        self.loss_function = loss_function

    def __call__(self, network):
        """
        Args:
            network: lasagne network



        Returns:
            theano compiled train function, theano compiled test function
        """
        logging.info(pprint.pformat(self.__dict__))
        # get loss and update expressions
        target_var = T.fmatrix('targets')  # fmatrix b/c it's a batch of float vectors
        input_var = lasagne.layers.get_all_layers(network)[0].input_var
        known_mask = T.gt(target_var, 0)
        loss = self.loss_function(network, target_var, known_mask, self.objective, self.regularization)
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = self.update['function'](loss, params, **self.update['kwargs'])

        # compile training function
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # get test loss
        test_loss = self.loss_function(network, target_var, known_mask, self.objective,
                                       regularization=None, deterministic=True)
        # compile test/validation function
        test_fn = theano.function([input_var, target_var], test_loss)

        return train_fn, test_fn


def squared_error_adam_train_test_func(learning_rate, beta1, beta2, epsilon, regularization):
    return GenericUpdateWithRegularization(
        objective=lasagne.objectives.squared_error,
        update=dict(
            function=lasagne.updates.adam,
            kwargs={'learning_rate': learning_rate,
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': epsilon}),
        regularization=regularization)


def squared_error_nesterov_momentum_train_test_func(learning_rate, momentum, regularization):
    return GenericUpdateWithRegularization(
        objective=lasagne.objectives.squared_error,
        update=dict(
            function=lasagne.updates.nesterov_momentum,
            kwargs={'learning_rate': learning_rate, 'momentum': momentum}),
        regularization=regularization)


def squared_err_nesterov_momentum_regularization(network, learning_rate, momentum, regularization):
    return squared_error_nesterov_momentum_train_test_func(learning_rate, momentum, regularization)(network)


def squared_error_adadelta_train_test_func(learning_rate, rho, epsilon, regularization):
    return GenericUpdateWithRegularization(
        objective=lasagne.objectives.squared_error,
        update=dict(
            function=lasagne.updates.adadelta,
            kwargs={'learning_rate': learning_rate,
                    'rho': rho,
                    'epsilon': epsilon}),
        regularization=regularization)


class ThresholdStochasticNegativeError(object):
    """
    Must be used with single value stochastic negatives. We determine where stoch negs are by finding
    where the target is equal to stochastic_negative_value.
    prediction[where stochastic_negative] = stochastic_negative_center
    objective(prediction, known) * (binary_mask_of_snegs_gt_threshold & binary_mask_of_non_snegs)
    """
    def __init__(self, objective, stochastic_negative_value, stochastic_negative_threshold, stochastic_negative_center):
        self.objective = objective
        self.stochastic_negative_value = lasagne.utils.as_theano_expression(stochastic_negative_value)
        self.stochastic_negative_threshold = lasagne.utils.as_theano_expression(stochastic_negative_threshold)
        self.stochastic_negative_center_value_diff = lasagne.utils.as_theano_expression(
            stochastic_negative_center - stochastic_negative_value)
        logging.info(pprint.pformat({'ThresholdStochasticNegativeError': self.__dict__}))

    def __call__(self, target_prediction, target_known):
        stochastic_negatives_mask = T.eq(target_known, self.stochastic_negative_value)
        gt_cutoff_mask = T.gt(target_prediction, self.stochastic_negative_threshold)
        stochastic_negatives_gt_cutoff_mask = stochastic_negatives_mask & gt_cutoff_mask
        target_prediction = target_prediction - stochastic_negatives_gt_cutoff_mask * self.stochastic_negative_center_value_diff
        loss = self.objective(target_prediction, target_known)
        return loss, stochastic_negatives_gt_cutoff_mask


class BatchAdjustedNegativeWeightLoss(object):
    """
    Weight loss of negative predictions to balance ratio of known positive and negative values in batch.
    """
    def __init__(self, objective, negative_threshold, allow_decreased_weights=False):
        self.objective = objective
        self.negative_threshold = lasagne.utils.as_theano_expression(lasagne.utils.floatX(negative_threshold))
        self.allow_decreased_weights = allow_decreased_weights
        logging.info(pprint.pformat({'BatchAdjustedNegativeWeightLoss': self.__dict__}))

    def __call__(self, target_prediction, target_known):
        known_values = T.gt(target_known, 0)
        known_negatives = known_values & T.le(target_known, self.negative_threshold)
        known_positives = T.gt(target_known, self.negative_threshold)
        negative_target_counts = T.sum(known_negatives, axis=0,
                                       dtype=theano.config.floatX,
                                       keepdims=True)
        positive_target_counts = T.sum(known_positives, axis=0,
                                       dtype=theano.config.floatX,
                                       keepdims=True)
        target_weight_with_nan = positive_target_counts / negative_target_counts
        # replace nan with zero
        negative_weights_targets = T.switch(T.eq(negative_target_counts, 0),
                                            negative_target_counts * lasagne.utils.floatX(0.),
                                            target_weight_with_nan)
        if not self.allow_decreased_weights:
            negative_weights_targets = T.switch(T.gt(negative_weights_targets, 1.),
                                                negative_weights_targets,
                                                lasagne.utils.floatX(1.))
        else:
            # replace zero with 1 where all known are negative
            negative_weights_targets = T.switch(T.eq(positive_target_counts, 0) & T.gt(negative_target_counts, 0),
                                                lasagne.utils.floatX(1.),
                                                negative_weights_targets)
        negative_weights = known_negatives * negative_weights_targets
        weights = T.switch(known_negatives, negative_weights, known_values)
        loss = self.objective(target_prediction, target_known)
        return loss * weights
