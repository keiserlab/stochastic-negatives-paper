import unittest
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne_nn.train as train


class TestBatchAdjustedNegativeWeightLossAllowDecreasedWeights(unittest.TestCase):

    def setUp(self):
        self.custom_set_up(allow_decreased_weights=True)

    def custom_set_up(self, allow_decreased_weights):
        self.allow_decreased_weights = allow_decreased_weights
        self.negative_threshold = 5.0
        p = self.negative_threshold + 1.
        n = self.negative_threshold - 1.
        t = self.negative_threshold
        _ = 0.
        self.knowns = np.array([
            [_, p, p, p, n, n, n, p],
            [_, p, p, p, n, n, n, p],
            [_, p, p, p, n, n, n, p],
            [_, _, n, p, _, p, n, t]], dtype=theano.config.floatX)
        _ = 0.
        p = 1.
        n = 1.
        u = 3.
        d = 1. / 3.
        self.expected_weights_no_downweighting = np.array([
            [_, p, p, p, n, n, n, p],
            [_, p, p, p, n, n, n, p],
            [_, p, p, p, n, n, n, p],
            [_, _, u, p, _, p, n, u]], dtype=theano.config.floatX)

        self.expected_weights_downweighting = np.array([
            [_, p, p, p, n, d, n, p],
            [_, p, p, p, n, d, n, p],
            [_, p, p, p, n, d, n, p],
            [_, _, u, p, _, p, n, u]], dtype=theano.config.floatX)
        self.expected_weights = (self.expected_weights_downweighting
                                 if self.allow_decreased_weights
                                 else self.expected_weights_no_downweighting)
        self.num_targets = self.knowns.shape[1]
        squared_err = lasagne.objectives.squared_error
        weighted_negatives_squared_err = train.BatchAdjustedNegativeWeightLoss(
            objective=squared_err,
            negative_threshold=self.negative_threshold,
            allow_decreased_weights=self.allow_decreased_weights)
        target_var = T.fmatrix('targets')
        known_values = T.gt(target_var, 0.)
        prediction = target_var + lasagne.utils.floatX(0.2)
        regular_loss = squared_err(prediction, target_var) * known_values
        weighted_loss = weighted_negatives_squared_err(prediction, target_var)
        weights = weighted_loss / regular_loss
        weights = T.switch(T.gt(regular_loss, 0), weights, lasagne.utils.floatX(0.))
        get_losses = theano.function([target_var],
                                     (prediction, weighted_loss,
                                      regular_loss, weights))
        (self.prediction, self.weighted_loss,
         self.regular_loss, self.weights) = get_losses(self.knowns)
        # self.weights = (self.weighted_loss / self.regular_loss)

    def test_no_nans_in_loss(self):
        self.assertEqual(np.isnan(self.weighted_loss).tolist(),
                         np.zeros_like(self.weighted_loss, dtype=bool).tolist())

    def test_no_inf_in_loss(self):
        self.assertEqual((np.isinf(self.weighted_loss) &
                         np.isneginf(self.weighted_loss)).tolist(),
                         np.zeros_like(self.weighted_loss, dtype=bool).tolist())

    def test_unknown_values_zero_weighted(self):
        col = 0
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())
        # self.assertTrue(np.all( == ))

    def test_no_known_negatives_positives_unweighted(self):
        col = 1
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def test_negatives_up_weighted(self):
        col = 2
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def test_target_with_all_positives_not_weighted(self):
        col = 3
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def no_known_positives_negatives_unweighted(self):
        col = 4
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def test_negatives_down_weighted_or_not(self):
        col = 5
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def test_target_with_all_negatives_not_weighted(self):
        col = 6
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())

    def test_threshold_treated_as_negative(self):
        col = 7
        self.assertEqual(self.weights[:, col].tolist(),
                         self.expected_weights[:, col].tolist())


class TestBatchAdjustedNegativeWeightLossDoNotAllowDecreasedWeights(
        TestBatchAdjustedNegativeWeightLossAllowDecreasedWeights):

    def setUp(self):
        self.custom_set_up(allow_decreased_weights=False)

if __name__ == '__main__':
    unittest.main()
