# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Top-level PyTorch trainer.

import os
import sling
import sling.flags as flags
import sys
import time
import torch

from functools import partial

sys.path.insert(0, "sling/nlp/parser/trainer")

import train_util as utils

from train_util import mem
from train_util import now
from train_util import Resources
from pytorch_modules import Sempar
from pytorch_modules import fstr

from corpora import Corpora

Var = torch.autograd.Variable

# Setting an explicit seed for the sake of determinism.
torch.manual_seed(1)


# Computes accuracy on the given dev set, using the given PyTorch Sempar module.
def dev_accuracy(commons_path, commons, dev_path, schema, tmp_folder, sempar):
  dev = Corpora(dev_path, commons, schema, gold=False, loop=False)
  print "Annotating dev documents", now(), mem()
  test_path = os.path.join(tmp_folder, "dev.annotated.rec")
  writer = sling.RecordWriter(test_path)
  count = 0
  start_time = time.time()
  for document in dev:
    state = sempar.forward(document, train=False)
    state.write()
    writer.write(str(count), state.encoded())
    count += 1
    if count % 100 == 0:
      print "  Annotated", count, "documents", now(), mem()
  writer.close()
  end_time = time.time()
  print "Annotated", count, "documents in", "%.1f" % (end_time - start_time), \
      "seconds", now(), mem()

  return utils.frame_evaluation(gold_corpus_path=dev_path, \
                                test_corpus_path=test_path, \
                                commons_path=commons_path)


# Implements loss-normalization policies for cascades.
#
# Recall that for a batch item I (i.e. a sentence), and a single gold action G
# in its gold sequence, there is a list L of (component, sub_action) pairs.
# Each such (c, s) pair incurs a loss from c's softmax layer during training.
# Typically the loss is a cross-entropy loss, which means that the loss ranges
# in [0, softmax_size(c)].
#
# A naive way to compute the loss for the entire batch is to calculate:
#   \sum_I \sum_{G \in I} \sum_{(c, s) \in G} Loss(c, s)
#
# But this is dominated by long batch items I, long gold translations (L),
# and bigger components (since the losses for them can be much larger).
#
# One way to be fair to all batch items and components is to a) first compute
# a BatchSize x NumComponents matrix M of losses, where (i, j) entry is the
# total loss incurred by batch item i in component j.
#
# M(i, j) = \sum_{G \in i} \sum_{(c, s) \in G: c = j} Loss(c, s)
#
# and have a corresponding counts matrix N:
# N(i, j) = \sum_{G \in i} \sum_{(c, s) \in G: c = j} 1
#
# Then compute the final normalized loss as:
#
# NormalizedLoss = \sum_{i, j} M(i, j) / (N(i, j) * SoftmaxSize(j))
# It can be further divided by BatchSize to get batch-size-independent losses.
#
# Another alternative is to compute:
#
# NormalizedLoss2 = \sum_j \sum_i M(i, j) / \sum_i (N(i, j) * SoftmaxSize(j))
# This need not be further divided by BatchSize.
#
# Note that if we only have one component (i.e. non-cascaded version), then
# NormalizedLoss = \sum_i M_i / N_i * (SoftmaxSize(0)), and
# NormalizedLoss2 = (\sum_i M_i) / \sum_i (N_i * SoftmaxSize(0)),
#
# i.e. NormalizedLoss2 is just a constant times our ordinary non-cascade loss.
class CascadeLoss:
  NORMALIZE_PER_COMPONENT_PER_ITEM = 0
  NORMALIZE_PER_COMPONENT_ACROSS_ITEMS = 1

  def __init__(self, normalizer, component_sizes, \
               divide_by_component_size, divide_by_batch_size):
    self.normalizer = normalizer
    self.divide_by_size = divide_by_component_size
    self.divide_by_batch_size = divide_by_batch_size

    if divide_by_batch_size and \
      normalizer == self.NORMALIZE_PER_COMPONENT_ACROSS_ITEMS:
      print "Note: divide_by_batch_size=True has no effect with "\
        "NORMALIZE_PER_COMPONENT_ACROSS_ITEMS policy"

    self.num_components = len(component_sizes)
    self.sizes = component_sizes
    self.loss = []
    self.counts = []

  # Resets loss computation for a new batch.
  def reset(self, batch_size):
    self.loss = [None] * batch_size
    self.counts = [None] * batch_size
    for i in xrange(batch_size):
      self.loss[i] = {}
      self.counts[i] = {}
      for j in xrange(self.num_components):
        self.loss[i][j] = Var(torch.Tensor([0.0]))
        self.counts[i][j] = 0


  # Adds a single component's loss on one sub-action to the overall loss.
  def add(self, item_index, component_index, loss):
    self.loss[item_index][component_index] += loss
    self.counts[item_index][component_index] += 1


  # Computes the final aggregated batch loss using specified normalization.
  def compute(self):
    loss = Var(torch.Tensor([0.0]))
    if self.normalizer == CascadeLoss.NORMALIZE_PER_COMPONENT_PER_ITEM:
      for i in xrange(len(self.loss)):
        for j in self.loss[i]:
          term = self.loss[i][j] / self.counts[i][j]
          if self.divide_by_size:
            term /= self.sizes[j]
          loss += term
      if self.divide_by_batch_size: loss /= len(self.loss)
    elif self.normalizer == CascadeLoss.NORMALIZE_PER_COMPONENT_ACROSS_ITEMS:
      for j in xrange(self.num_components):
        component_loss = Var(torch.Tensor([0.0]))
        component_count = 0
        for i in xrange(len(self.loss)):
          if j in self.loss[i]:
            component_loss += self.loss[i][j]
            component_count += self.counts[i][j]
        component_loss /= component_count
        if self.divide_by_size:
          component_loss /= self.sizes[j]
        loss += component_loss
    return loss


# A trainer reads one example at a time, till a count of num_examples is
# reached. For each example it computes the loss.
# After every 'batch_size' examples, it computes the gradient and applies
# it, with optional gradient clipping.
class Trainer:
  # Training hyperparameters.
  class Hyperparams:
    def __init__(self, args):
      # Sets various hyperparameters from 'args'.
      self.alpha = args.learning_rate
      self.batch_size = args.batch_size
      self.num_examples = args.train_steps * self.batch_size
      self.report_every = args.report_every * self.batch_size
      self.l2_coeff = args.l2_coeff
      self.gradient_clip = args.gradient_clip_norm
      self.optimizer = args.learning_method
      self.adam_beta1 = args.adam_beta1
      self.adam_beta2 = args.adam_beta2
      self.adam_eps = args.adam_eps
      self.moving_avg = args.use_moving_average
      self.moving_avg_coeff = args.moving_average_coeff
      for k, v in self.__dict__.iteritems():
        assert v is not None, "Hyperparameter %r not set" % k


    # Returns a string representation of all hyperparameters.
    def __str__(self):
      return str(self.__dict__)


  # Instantiates the trainer with the given model, optional evaluator,
  # and hyperparameters.
  def __init__(self, sempar, hyperparams, evaluator=None, \
               output_file_prefix=None):
    self.model = sempar
    self.evaluator = evaluator
    self.hyperparams = hyperparams

    if hyperparams.optimizer == "sgd":
      self.optimizer = torch.optim.SGD(
        sempar.parameters(),
        lr=self.hyperparams.alpha,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False)
    elif hyperparams.optimizer == "adam":
      self.optimizer = torch.optim.Adam(
          sempar.parameters(), lr=hyperparams.alpha, weight_decay=0, \
              betas=(hyperparams.adam_beta1, hyperparams.adam_beta2), \
              eps=hyperparams.adam_eps)
    else:
      raise ValueError('Unknown learning method: %r' % hyperparams.optimizer)

    num_params = 0
    for name, p in sempar.named_parameters():
      if p.requires_grad:
        print name, ":", p.size()
        num_params += torch.numel(p)
    print "Number of parameters:", num_params

    self.count = 0
    self.last_eval_count = 0

    self.current_batch_size = 0
    self.batch_loss = CascadeLoss(
      CascadeLoss.NORMALIZE_PER_COMPONENT_ACROSS_ITEMS, \
      sempar.component_sizes(),
      divide_by_component_size=False,
      divide_by_batch_size=True)
    self._reset()

    self.checkpoint_metrics = []
    self.best_metric = None
    self.output_file_prefix = output_file_prefix

    # Exponential moving average clones.
    self.averages = {}
    if hyperparams.moving_avg:
      for name, p in sempar.named_parameters():
        if p.requires_grad:
          self.averages[name] = p.data.clone()


  # Resets the state for a new batch.
  def _reset(self):
    self.current_batch_size = 0
    self.batch_loss.reset(self.hyperparams.batch_size)
    self.optimizer.zero_grad()


  # Processes a single given example.
  def process(self, example):
    losses = self.model.forward(example, train=True)
    for loss, component_index in losses:
      self.batch_loss.add(self.current_batch_size, component_index, loss)

    self.current_batch_size += 1
    self.count += 1
    if self.current_batch_size == self.hyperparams.batch_size:
      self.update()
    if self.count % self.hyperparams.report_every == 0:
      self.evaluate()


  # Clips the gradients separately for each PyTorch Parameter.
  def clip_gradients(self):
    if self.hyperparams.gradient_clip > 0.0:
      for p in self.model.parameters():
        torch.nn.utils.clip_grad_norm([p], self.hyperparams.gradient_clip)


  # Performs a gradient update.
  def update(self):
    if self.current_batch_size > 0:
      start = time.time()
      torch_loss = self.batch_loss.compute()

      # Add the regularization penalty to the batch loss.
      l2 = Var(torch.Tensor([0.0]))
      if self.hyperparams.l2_coeff > 0.0:
        for p in self.model.regularized_params:
          l2 += 0.5 * self.hyperparams.l2_coeff * torch.sum(p * p)
        torch_loss += l2

      torch_loss /= 3.0  # for parity with TF
      value = torch_loss.data[0]

      # Compute gradients.
      torch_loss.backward()

      # Clip them.
      self.clip_gradients()

      # Apply them.
      self.optimizer.step()

      # Done for this batch, prepare for the next one.
      self._reset()
      end = time.time()
      num_batches = self.count / self.hyperparams.batch_size

      if self.hyperparams.moving_avg:
        # Update the moving averages.
        # Use a more conservative decay factor in the first few batches.
        decay = self.hyperparams.moving_avg_coeff
        decay2 = (1.0 + num_batches) / (10.0 + num_batches)
        if decay > decay2: decay = decay2

        for name, p in self.model.named_parameters():
          if p.requires_grad and name in self.averages:
            diff = (self.averages[name] - p.data) * (1 - decay)
            self.averages[name].sub_(diff)

      print "BatchLoss after", "(%d" % num_batches, \
          "batches =", self.count, "examples):", value, \
          " incl. L2=", fstr(l2 / 3.0), \
          "(%.1f" % (end - start), "secs)", now(), mem()


  # Swaps model parameters with their moving average counterparts.
  # This just swaps pointers to data, so is very cheap.
  def _swap_with_ema_parameters(self):
    if not self.hyperparams.moving_avg: return
    for name, p in self.model.named_parameters():
      if name in self.averages:
        tmp = self.averages[name]
        self.averages[name] = p.data
        p.data = tmp


  # Runs the current model on the evaluator.
  def evaluate(self):
    if self.evaluator is not None:
      if self.count != self.last_eval_count:
        # Use average parameters if available.
        self._swap_with_ema_parameters()

        metrics = self.evaluator(self.model)
        self.checkpoint_metrics.append((self.count, metrics))
        eval_metric = metrics["eval_metric"]
        print "Eval metric after", self.count, " examples:", eval_metric

        if self.output_file_prefix is not None:
          if self.best_metric is None or self.best_metric < eval_metric:
            self.best_metric = eval_metric
            best_model_file = self.output_file_prefix + ".best.model"
            torch.save(self.model.state_dict(), best_model_file)
            print "Updating best model at", best_model_file

            best_flow_file = self.output_file_prefix + ".best.flow"
            self.model.write_flow(best_flow_file)
            print "Updating best flow at", best_flow_file

        self.last_eval_count = self.count

        # Swap back.
        self._swap_with_ema_parameters()


  # Trains the model using 'corpora'.
  def train(self, corpora):
    corpora.rewind()
    corpora.set_loop(True)
    for document in corpora:
      if self.count >= self.hyperparams.num_examples:
        break
      self.process(document)

    # Process the partial batch (if any) at the end, and evaluate one last time.
    self.update()
    self.evaluate()


def check_present(args, ls):
  for x in ls:
    val = getattr(args, x, None)
    assert val is not None, "%r should be present" % x
    if type(val) is str:
      assert val != "", "%r should be set" % x


def train(args):
  check_present(
      args, ["commons", "train_corpus", "output_folder", "dev_corpus"])
  resources = utils.Resources()
  resources.load(commons_path=args.commons,
                 train_path=args.train_corpus,
                 word_embeddings_path=args.word_embeddings)

  sempar = Sempar(resources.spec)
  sempar.initialize()

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      args.dev_corpus,
                      resources.schema,
                      tmp_folder)

  output_file_prefix = os.path.join(args.output_folder, "pytorch")
  hyperparams = Trainer.Hyperparams(args)
  print "Using hyperparameters:", hyperparams

  trainer = Trainer(sempar, hyperparams, evaluator, output_file_prefix)
  trainer.train(resources.train)


def evaluate(args):
  check_present(args, ["commons", "train_corpus", "dev_corpus", "model_file"])
  resources = utils.Resources()
  resources.load(commons_path=args.commons,
                 train_path=args.train_corpus,
                 word_embeddings_path=args.word_embeddings)

  sempar = Sempar(resources.spec)
  sempar.load_state_dict(torch.load(args.model_file))

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      args.dev_corpus,
                      resources.schema,
                      tmp_folder)
  metrics = evaluator(sempar)
  print "Eval metric", metrics["eval_metric"]


if __name__ == '__main__':
  utils.setup_training_flags(flags)
  flags.define('--mode',
               help='Mode: train or evaluate',
               default='train',
               type=str)
  flags.define('--model_file',
               help='PyTorch model file',
               default='',
               type=str)
  flags.parse()

  if flags.arg.mode == "train":
    train(flags.arg)
  elif flags.arg.mode == "evaluate":
    evaluate(flags.arg)
  else:
    raise ValueError('Unknowm mode %r' % flags.arg)
