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

import inspect
import sling

from action import Action
from action_table import Actions

"""A cascade is a mechanism to output a parser action in a staged manner.
Each stage of the cascade is handled by a Delegate, which either computes
a piece of the eventual action, or delegates the computation to another
delegate, or both.

This mechanism allows each delegate to focus on a small part of the action,
thus allowing scalability (e.g. smaller softmax layer per delegate), as well
as generality (since custom delegates can be added easily).
"""


"""Cascade delegate interface."""
class Delegate(object):
  """Type of the delegate."""
  SOFTMAX = 0     # For multi-class classification
  RANKING = 1     # For ranking-based delegates

  def __init__(self, t=None):
    self.type = t
    self.model = None
    self.lossfn = None
    self.actions = None

  """Sets the loss function for the delegate."""
  def set_loss(self, lossfn):
    self.lossfn = lossfn

  """Sets the implementation-specific model function for the delegate."""
  def set_model(self, model):
    self.model = model

  """Delegate interface."""

  """Prepares the delegate. Called before anything else.
  'cascade' is the cascade that this delegate is a part of, and 'actions'
  is the action table for the parser."""
  def build(self, cascade, actions):
    pass

  def translate(self, action, output):
    pass

  def loss(self, state, ff_hidden, gold):
    pass

  def predict(self, state, previous_action, ff_hidden):
    pass

  def as_frame(self, frame):
    pass


"""Delegate that uses a softmax layer to decide what action to output."""
class SoftmaxDelegate(Delegate):
  def __init__(self):
    super(SoftmaxDelegate, self).__init__(Delegate.SOFTMAX)
    self.softmax_size = None

  """Returns the size of the softmax layer."""
  def size(self):
    return self.softmax_size

  """Returns the integer index of 'action' in the softmax layer."""
  def index(self, action):
    pass

  """Returns the action corresponding to 'index' in the softmax layer."""
  def action(index, previous_action=None):
    pass

  def loss(self, state, ff_hidden, gold):
    logits = self.model(ff_hidden, train=True)
    gold_index = self.index(gold)
    return self.lossfn(logits, gold_index)

  def predict(self, state, previous_action, ff_hidden):
    best_index = self.model(ff_hidden, train=False)
    return self.action(best_index, previous_action)

  def as_frame(self, frame):
    actions = frame.store().array(self.size())
    for i in xrange(self.size()):
      action = self.action(i, previous_action=None)
      actions[i] = action.as_frame(frame.store())
    frame["actions"] = actions


class FlatDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.actions = actions
    self.softmax_size = actions.size()

  def translate(self, action, output):
    output.append(action)
      
  def index(self, action):
    return self.actions.index(action)
    
  def action(self, index, previous_action):
    return self.actions.action(index)


class ShiftOrNotDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.softmax_size = 2
    self.shift = Action(Action.SHIFT)
    self.not_shift = Action(Action.CASCADE)
    assert self is cascade.delegates[0]  # Should be the top delegate
    assert cascade.size() > 1
    # Assume we will always cascade to the next delegate.
    self.not_shift.delegate = 1

  def translate(self, action, output):
    if action.type == Action.SHIFT:
      output.append(self.shift)
    else:
      output.append(self.not_shift)
      
  def index(self, action):
    return 0 if action.type == Action.SHIFT else 1
    
  def action(self, index, previous_action):
    return self.shift if index == 0 else self.not_shift


class NotShiftDelegate(SoftmaxDelegate):
  def __init__(self):
    super(NotShiftDelegate, self).__init__()

  def build(self, cascade, actions):
    self.shift = actions.shift()
    self.actions = actions
    self.softmax_size = self.actions.size() - 1

  def translate(self, action, output):
    output.append(action)

  def index(self, action):
    i = self.actions.index(action)
    if i > self.shift: i -= 1
    return i

  def action(self, index, previous_action):
    if index >= self.shift: index += 1
    return self.actions.action(index)


"""Returns whether 'action' EVOKEs a Propbank frame."""
def is_pbevoke(action):
  return action.type == Action.EVOKE and action.label.id.startswith("/pb/")


class ExceptPropbankEvokeDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.table = Actions()
    for action in actions.table:
      if action.type != Action.SHIFT and not is_pbevoke(action):
        self.table.add(action)
    self.softmax_size = self.table.size() + 1
    self.pb_index = self.table.size()

    self.pb_action = Action(Action.CASCADE)
    self.pb_action.delegate = cascade.index_of("PropbankEvokeDelegate")

  def translate(self, action, output):
    if is_pbevoke(action):
      output.append(self.pb_action)
    else:
      output.append(action)

  def index(self, action):
    if action.is_cascade() and action.delegate == self.pb_action.delegate:
      return self.pb_index
    return self.table.index(action)

  def action(self, index, previous_action):
    if index == self.pb_index:
      return self.pb_action
    return self.table.action(index)


class PropbankEvokeDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.table = Actions()
    for action in actions.table:
      if is_pbevoke(action): self.table.add(action)
    self.softmax_size = self.table.size()

  def translate(self, action, output):
    output.append(action)

  def index(self, action):
    return self.table.index(action)

  def action(self, index, previous_action):
    return self.table.action(index)


class EvokeLengthDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.table = Actions()
    self.cascade_actions = {}
    evoke_type_delegate = cascade.index_of("EvokeTypeDelegate")
    for action in actions.table:
      if action.type == Action.EVOKE:
        if action.length not in self.cascade_actions:
          cascade_action = Action(Action.CASCADE)
          cascade_action.delegate = evoke_type_delegate
          cascade_action.length = action.length
          self.cascade_actions[action.length] = cascade_action
        else:
          cascade_action = self.cascade_actions[action.length]
        action = cascade_action
      self.table.add(action)
    self.softmax_size = self.table.size()

  def translate(self, action, output):
    if action.type == Action.EVOKE:
      output.append(self.cascade_actions[action.length])
    else:
      output.append(action)

  def index(self, action):
    return self.table.index(action)

  def action(self, index, previous_action):
    return self.table.action(index)


class EvokeTypeDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.types = {}
    self.index_to_types = []
    for action in actions.table:
      if action.type != Action.EVOKE: continue
      if action.label not in self.types:
        index = len(self.types)
        self.index_to_types.append(action.label)
        self.types[action.label] = index
    self.softmax_size = len(self.types)

  def translate(self, action, output):
    assert action.type == Action.EVOKE
    output.append(action)

  def index(self, action):
    return self.types[action.label]

  def action(self, index, previous_action):
    action = Action(Action.EVOKE)
    action.label = self.index_to_types[index]
    if previous_action is not None:
      action.length = previous_action.length
    return action


"""Cascade interface."""
class Cascade(object):
  def __init__(self, actions):
    self.actions = actions
    self.delegates = []

  def add(self, delegate):
    self.delegates.append(delegate)

  
  def initialize(self, delegate_classes):
    for delegate in delegate_classes:
      self.add(delegate())
    for delegate in self.delegates:
      delegate.build(self, self.actions)

    
  def index_of(self, delegate):
    if inspect.isclass(delegate):
      for index, d in enumerate(self.delegates):
        if d.__class__ is delegate: return index
      raise ValueError("Can't find delegate %r" % delegate.__name__)
    elif type(delegate) is str:
      for index, d in enumerate(self.delegates):
        if d.__class__.__name__ == delegate: return index
      raise ValueError("Can't find delegate %r" % delegate)
    elif isinstance(delegate, Delegate):
      for index, d in enumerate(self.delegates):
        if d is delegate: return index
      raise ValueError("Can't find delegate %r" % delegate.__class__.__name__)
    raise ValueError("Can't handle delegate", delegate)

  def size(self):
    return len(self.delegates)

  """Returns the next delegate id, assuming 'action' to be a CASCADE."""
  def next(self, action, delegate_index):
    return action.delegate

  def translate(self, sequence):
    output = []
    for action in sequence:
      delegate_index = 0
      while True:
        self.delegates[delegate_index].translate(action, output)
        if output[-1].is_cascade():
          delegate_index = self.next(output[-1], delegate_index)
        else:
          break
    return output

  def loss(self, delegate_index, state, ff_hidden, gold):
    return self.delegates[delegate_index].loss(state, ff_hidden, gold)

  def predict(self, delegate_index, state, previous_action, ff_hidden):
    return self.delegates[delegate_index].predict(
      state, previous_action, ff_hidden)

  def __repr__(self):
    s = self.__class__.__name__ + "(" + str(len(self.delegates)) + " delegates)"
    for d in self.delegates:
      s += "\n  " + d.__class__.__name__
      if d.type == Delegate.SOFTMAX:
        s += " (softmax size=" + str(d.size()) + ")"
      else:
        s += " (ranking)"
    return s

  def as_frame(self, store):
    frame = store.frame({"name": self.__class__.__name__})
    delegates = store.array(self.size())
    for index, delegate in enumerate(self.delegates):
      d = store.frame({"name": delegate.__class__.__name__, "index": index})
      delegate.as_frame(d)
      delegates[index] = d
    frame["delegates"] = delegates
    return frame


class FlatCascade(Cascade):
  def __init__(self, actions):
    super(FlatCascade, self).__init__(actions)
    self.initialize([FlatDelegate])


class ShiftCascade(Cascade):
  def __init__(self, actions):
    super(ShiftCascade, self).__init__(actions)
    self.initialize([ShiftOrNotDelegate, NotShiftDelegate])


class ShiftPropbankEvokeCascade(Cascade):
  def __init__(self, actions):
    super(ShiftPropbankEvokeCascade, self).__init__(actions)
    self.initialize(
      [ShiftOrNotDelegate, ExceptPropbankEvokeDelegate, PropbankEvokeDelegate])


# Delegate 0: SHIFT or not (binary)
# Delegate 1: All non-EVOKE actions, plus only the EVOKE length.
# Delegate 2: Just the EVOKE type.
class ShiftEvokeCascade(Cascade):
  def __init__(self, actions):
    super(ShiftEvokeCascade, self).__init__(actions)
    self.initialize(
      [ShiftOrNotDelegate, EvokeLengthDelegate, EvokeTypeDelegate])


def print_cost_estimates():
  import train_util as utils
  from corpora import Corpora
  path = "/home/grahul/sempar_ontonotes/"
  resources = utils.Resources()
  resources.load(path + "commons", path + "train.rec")
  resources.train.rewind()
  resources.train.set_gold(True)

  actions = resources.spec.actions
  cascades = [c(actions) for c in \
    [FlatCascade, ShiftCascade, ShiftPropbankEvokeCascade, ShiftEvokeCascade]]
  costs = [0] * len(cascades)
  for document in resources.train:
    gold = document.gold
    for index, cascade in enumerate(cascades):
      cascade_gold = cascade.translate(gold)
      delegate = 0
      cost = 0
      for cg in cascade_gold:
        cost += cascade.delegates[delegate].size()
        if cg.is_cascade():
          delegate = cg.delegate
        else:
          delegate = 0
      costs[index] += cost

  for cost, cascade in zip(costs, cascades):
    print "\n", cascade.__class__.__name__, "cost =", cost, "\n", cascade

#print_cost_estimates()
