# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""
Implementation of the disentanglement metric from the FactorVAE paper.
Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
"""

import logging
from tqdm import tqdm

from disent.dataset.ground_truth_data.base_data import GroundTruthData
from disent.dataset.single import GroundTruthDataset
from disent.metrics import utils
import numpy as np
from disent.util import to_numpy


# ========================================================================= #
# factor_vae                                                                #
# ========================================================================= #


def compute_factor_vae(
        ground_truth_data: GroundTruthDataset,
        representation_function: callable,
        batch_size: int = 64,
        num_train: int = 10000,
        num_eval: int = 5000,
        num_variance_estimate: int = 10000,
        show_progress=False,
):
    """
    Computes the FactorVAE disentanglement metric.

    Algorithm Description (Excerpt from paper):
    =====================

    1. Choose a factor k
    2. generate data with this factor fixed but all other factors varying randomly;
    3. obtain their representations;
    4. normalise each dimension by its empirical standard deviation over the full data
      (or a large enough random subset);
    5. take the empirical variance in each dimension of these normalised representations.
    6. Then the index of the dimension with the lowest variance and the target index k
       provide one training input/output example for the classifier (see bottom of Figure 2).

    # ---------------------------------------------------------------------- #
    | Thus if the representation is perfectly disentangled, the empirical    |
    | variance in the dimension corresponding to the fixed factor will be 0. |
    # ---------------------------------------------------------------------- #

    - We normalise the representations (above) so that the arg min is invariant to rescaling
      of the representations in each dimension.

    - Since both inputs and outputs lie in a discrete space, the optimal
         classifier is the majority-vote classifier (see Appendix B for details),
         and the metric is the error rate of the classifier.

    The resulting classifier is a deterministic function of the training data, hence there are no optimisation hyperparameters to tune.
    We also believe that this metric is conceptually simpler and more natural than the previous one.
    Most importantly, it circumvents the failure mode of the earlier metric, since the classifier needs to see the lowest variance in a latent dimension for a given factor to classify it correctly

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      batch_size: Number of points to be used to compute the training_sample.
      num_train: Number of points used for training.
      num_eval: Number of points used for evaluation.
      num_variance_estimate: Number of points used to estimate global variances.
      show_progress: If a tqdm progress bar should be shown
    Returns:
      Dictionary with scores:
        train_accuracy: Accuracy on training set.
        eval_accuracy: Accuracy on evaluation set.
    """

    logging.info("Computing global variances to standardise.")
    global_variances = _compute_variances(ground_truth_data, representation_function, num_variance_estimate)
    active_dims = _prune_dims(global_variances)
    scores_dict = {}

    if not active_dims.any():
        scores_dict["factor_vae.train_accuracy"] = 0.
        scores_dict["factor_vae.eval_accuracy"] = 0.
        scores_dict["factor_vae.num_active_dims"] = 0
        return scores_dict

    logging.info("Generating training set.")
    training_votes = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_train, global_variances, active_dims, show_progress=show_progress)
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    logging.info("Evaluate training set accuracy.")
    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    logging.info("Generating evaluation set.")
    eval_votes = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_eval, global_variances, active_dims, show_progress=show_progress)

    logging.info("Evaluate evaluation set accuracy.")
    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

    logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
    scores_dict["factor_vae.train_accuracy"] = train_accuracy
    scores_dict["factor_vae.eval_accuracy"] = eval_accuracy
    scores_dict["factor_vae.num_active_dims"] = len(active_dims)

    return scores_dict

def _prune_dims(variances, threshold=0.):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(
        ground_truth_data: GroundTruthData,
        representation_function: callable,
        batch_size: int,
        eval_batch_size: int = 64
):
    """Computes the variance for each dimension of the representation.
    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and outputs a representation.
      batch_size: Number of points to be used to compute the variances.
      eval_batch_size: Batch size used to eval representation.
    Returns:
      Vector with the variance of each dimension.
    """
    observations = ground_truth_data.sample_observations(batch_size)
    observations = observations.cuda()
    representations = to_numpy(utils.obtain_representation(observations, representation_function, eval_batch_size))
    representations = np.transpose(representations)
    assert representations.shape[0] == batch_size
    return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(
        ground_truth_data: GroundTruthData,
        representation_function: callable,
        batch_size: int,
        global_variances: np.ndarray,
        active_dims: list,
) -> (int, int):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and
        outputs a representation.
      batch_size: Number of points to be used to compute the training_sample.
      global_variances: Numpy vector with variances for all dimensions of representation.
      active_dims: Indexes of active dimensions.
    Returns:
      factor_index: Index of factor coordinate to be used.
      argmin: Index of representation coordinate with the least variance.
    """
    # Select random coordinate to keep fixed.
    factor_index = np.random.randint(ground_truth_data.num_factors)
    # Sample two mini batches of latent variables.
    factors = ground_truth_data.sample_factors(batch_size)
    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]
    # Obtain the observations.
    observations = ground_truth_data.sample_observations_from_factors(factors).cuda()
    representations = to_numpy(representation_function(observations))
    local_variances = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
    return factor_index, argmin


def _generate_training_batch(
        ground_truth_data: GroundTruthData,
        representation_function: callable,
        batch_size: int,
        num_points: int,
        global_variances: np.ndarray,
        active_dims: list,
        show_progress=False,
):
    """Sample a set of training samples based on a batch of ground-truth data.
    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      batch_size: Number of points to be used to compute the training_sample.
      num_points: Number of points to be sampled for training set.
      global_variances: Numpy vector with variances for all dimensions of representation.
      active_dims: Indexes of active dimensions.
    Returns:
      (num_factors, dim_representation)-sized numpy array with votes.
    """
    votes = np.zeros((ground_truth_data.num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in tqdm(range(num_points), disable=(not show_progress)):
        factor_index, argmin = _generate_training_sample(ground_truth_data, representation_function, batch_size, global_variances, active_dims)
        votes[factor_index, argmin] += 1
    return votes

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
