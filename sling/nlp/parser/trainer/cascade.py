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

"""Cascade and cascade component interface."""

import sling

from action import Action
from action_table import Actions

""" Interface for a single component in a cascade.

A CascadeComponent takes an under-construction action as input and adds
more fields to it. The interfaces specifies the following:
- The actions it can take as an input.
- A range of output indices that governs the component's softmax size.
- Given an under-construction action, a parser state, and sub-action scores,
  which highest-scoring sub-action can be tacked on to the action.
- Given an under construction action and a component-specific sub-action,
  how to modify the input action.
"""
class CascadeComponent(object):
  """Constructs the cascade component using the given global action table."""
  def __init__(self, actions):
    self.actions = actions


  """Returns whether the component takes global action 'action' as input."""
  def handles(self, action):
    pass


  """Returns the output index to which the global action 'action' will be
  mapped. 'action' is a valid input action for the component."""
  def index(self, action):
    pass

  """Given a parser state, an under-construction action 'action_so_far',
  and the indices of the top-k sub-actions, returns the top-k sub-action
  that is valid w.r.t. 'state' and 'action_so_far'.
  """
  def best(self, state, action_so_far, topk_actions):
    pass


  """Modifies 'action_so_far' with the sub-action 'sub_action'."""
  def apply(self, action_so_far, sub_action):
    pass


  """Utility methods."""
  def is_pbevoke(self, action):
    return action.type == Action.EVOKE and action.label.id.startswith("/pb/")


  """Returns whether global action 'index' is allowed by 'state'."""
  def allowed(self, state, index):
    return not self.actions.disallowed[index] and state.is_allowed(index)


"""A top component is first in a cascade. By definition, it covers all global
actions."""
class TopComponent(CascadeComponent):
  def handles(self, action):
    return True


"""A flat component has one output index per global action. It is also a
top component."""
class FlatComponent(TopComponent):
  """Maps each global action to its own output index."""
  def index(self, action):
    return self.actions.indices.get(action)


  """Returns the action that is allowed per the state and the action table."""
  def best(self, state, action_so_far, topk_actions):
    top = None
    for a in topk_actions:
      if self.allowed(state, a):
        top = a
        break

    return top


  """Returns a copy of 'sub_action'."""
  def apply(self, action_so_far, sub_action):
    self.actions.action(sub_action).copy_to(action_so_far)


"""Top component that decides whether to shift or not."""
class ShiftOrNotComponent(TopComponent):
  SHIFT = 0
  NOT_SHIFT = 1

  """Maps SHIFT to 0, everything else to 1."""
  def index(self, action):
    if action.is_shift():
      return ShiftOrNotComponent.SHIFT
    return ShiftOrNotComponent.NOT_SHIFT


  """Returns SHIFT if it's the best and the state can SHIFT, else NOT_SHIFT."""
  def best(self, state, action_so_far, topk_actions):
    top = topk_actions[0]
    if top == ShiftOrNotComponent.SHIFT and state.can_shift():
      return top
    return ShiftOrNotComponent.NOT_SHIFT

  """Does nothing if we don't have to SHIFT."""
  def apply(self, action_so_far, sub_action):
    if sub_action == ShiftOrNotComponent.SHIFT:
      action_so_far.type = Action.SHIFT


"""Leaf component that decides what to do if we are not shifting. Paired
with ShiftOrNotComponent."""
class NonShiftComponent(CascadeComponent):
  def __init__(self, actions):
    super(NonShiftComponent, self).__init__(actions)
    self.shift = actions.shift()


  """Handles only non-SHIFT actions."""
  def handles(self, action):
    return not action.is_shift()


  """Every non-shift action gets its own output index.
  The index is just the global index after accounting for SHIFT's index."""
  def index(self, action):
    i = self.actions.index(action)
    if i < self.shift:
      return i
    return i - 1


  """Returns global action index for 'local_index'."""
  def global_action_id(self, local_index):
    if local_index >= self.shift: return local_index + 1
    return local_index


  """Returns best allowed sub-action for 'state'."""
  def best(self, state, action_so_far, topk_actions):
    for a in topk_actions:
      g = self.global_action_id(a)
      if self.allowed(state, g): return a

    return None


  """Applies the global equivalent of the given action to 'action_so_far'."""
  def apply(self, action_so_far, sub_action):
    g = self.global_action_id(sub_action)
    self.actions.action(g).copy_to(action_so_far)


"""Component that handles all actions except SHIFT as inputs.
- Maps all Propbank EVOKE actions to a single index at the end.
- Maps all other actions to individual indices.
"""
class ExceptShiftAndPropbankEvokeComponent(CascadeComponent):
  def __init__(self, actions):
    super(ExceptShiftAndPropbankEvokeComponent, self).__init__(actions)

    # Component-specific action-table.
    self.table = Actions()
    for action in self.actions.table:
      if self.handles(action) and not self.is_pbevoke(action):
        self.table.add(action)

    # Propbank evokes are mapped to the last index.
    self.pb_index = self.table.size()


  """Handles all actions except SHIFT as input."""
  def handles(self, action):
    return not action.is_shift()


  """Returns the index for 'action'."""
  def index(self, action):
    if not self.is_pbevoke(action):
      return self.table.index(action)
    return self.pb_index


  """Returns the highest scoring allowed sub-action for 'state'."""
  def best(self, state, action_so_far, topk_actions):
    for a in topk_actions:
      # If Propbank EVOKE is being considered, we shouldn't be at the end.
      if a == self.pb_index and state.current < state.end: return a

      # STOP is only permitted at the end.
      action = self.table.action(a)
      if action.is_stop() and state.current == state.end: return a

      # Everything else is allowed if the state permits it.
      global_id = self.actions.index(action)
      if self.allowed(state, global_id): return a

    return None


  """Applies the given sub-action. Does nothing if it is a Propbank EVOKE."""
  def apply(self, action_so_far, sub_action):
    if sub_action != self.pb_index:
      self.table.action(sub_action).copy_to(action_so_far)


"""Component that only handles Propbank EVOKEs, mapping each of them to
separate indices.
"""
class PropbankEvokeComponent(CascadeComponent):
  def __init__(self, actions):
    super(PropbankEvokeComponent, self).__init__(actions)

    # Make an action table for just the Propbank evokes.
    self.table = Actions()
    for action in self.actions.table:
      if self.handles(action): self.table.add(action)

  """Returns true only for Propbank evokes."""
  def handles(self, action):
    return self.is_pbevoke(action)


  """Returns the output index for 'action', which should be a Propbank evoke."""
  def index(self, action):
    return self.table.index(action)


  """Returns the best Propbank evoke which is allowed at 'state', else None."""
  def best(self, state, action_so_far, topk_actions):
    if state.current >= state.end: return None

    # Predicate should match the current word, e.g. /pb/eat-01 matches "eat".
    word = state.current_word().lower() + "-"
    for i in topk_actions:
      action = self.table.action(i).label.id[4:]
      if action.startswith(word): return i
    return None


  """Applies the given Propbank evoke to 'action_so_far'."""
  def apply(self, action_so_far, sub_action):
    self.table.action(sub_action).copy_to(action_so_far)


"""A Cascade is a DAG hierarchy of components.

It is initialized with a list of components, and the hierarchy edges
between the components are specified via connect() calls.
"""
class Cascade(object):
  def __init__(self, components, actions):
    self.name = self.__class__.__name__
    self.actions = actions
    self.components = components

    # Initialize the hierarchy as empty.
    self.children = []  # Index i -> Child(ren) components of component i
    for _ in components: self.children.append({})

    # Index i -> Softmax size for component i.
    self.sizes = [0] * self.size()

    # Index i -> Actions funnelled through component i.
    self.inputs = [0] * self.size()

    # Global action index i -> Softmax cost of computing global action i.
    self.costs = [0] * self.actions.size()


  """Sets the name of the component."""
  def set_name(self, name):
    self.name = name


  """Returns the name of the component."""
  def get_name(self):
    return self.name


  """Adds a link to the hierarchy.
  - 'parent' is the index of the parent component.
  - 'child' is the index of the child component.
  - 'index' is the output index of parent that will be fed to the child.

  The semantics is that if the parent decides on a choice 'index', then
  the choice is further propagated to the child, e.g. if ShiftOrNotComponent
  decides to NOT_SHIFT, it will be further propagated to NonShiftComponent.
  """
  def connect(self, parent, index, child):
    assert index not in self.children[parent], (parent, index)
    self.children[parent][index] = child


  """Returns the index of the child component (or None) that should
  be invoked if 'parent' makes an output choice of 'index'.
  """
  def next(self, parent, index):
    return self.children[parent].get(index, None)


  """Returns whether component 'component_index' has no children.
  """
  def is_leaf(self, component_index):
    return len(self.children[component_index]) == 0


  """Initializes all components, computes cost estimates, and softmax sizes."""
  def build(self, print_cost_estimate=False):
    # Compute component sizes.
    for action in self.actions.table:
      level = 0
      while level is not None:
        c = self.components[level]
        assert c.handles(action), (c.__class__.__name__, str(action))
        index = c.index(action)
        if self.sizes[level] <= index:
          self.sizes[level] = index + 1
        level = self.next(level, index)

    # Compute cost estimates.
    for i, action in enumerate(self.actions.table):
      level = 0
      while level is not None:
        action_count = self.actions.counts[i]
        self.inputs[level] += action_count
        self.costs[i] += action_count * self.sizes[level]
        index = self.components[level].index(action)
        level = self.next(level, index)

    # Print combined cost estimates for all actions.
    if print_cost_estimate:
      print self.get_name(), "total cost =", sum(self.costs)
      for i in xrange(self.size()):
        print "  Component", i, self.components[i].get_name(), \
          "softmax size =", self.sizes[i], ", #inputs =", self.inputs[i]


  """Returns the number of components in the hierarchy."""
  def size(self):
    return len(self.components)


  """Returns the softmax size of component 'index'."""
  def component_size(self, index):
    return self.sizes[index]


  """Translates 'gold' action to a list of (component, sub_action) pairs."""
  def translate(self, gold):
    output = []
    level = 0
    while level is not None:
      index = self.components[level].index(gold)
      output.append((level, index))
      level = self.next(level, index)

    return output


  """Predicts the best allowed sub-action for component 'level'.
  Returns (next component index, best sub-action index), with
  None wherever applicable.
  """
  def predict(self, level, state, action_so_far, topk_indices):
    c = self.components[level]
    best = c.best(state, action_so_far, topk_indices)
    if best is None:
      return (None, None)

    c.apply(action_so_far, best)
    if self.is_leaf(level):
      level = None
    else:
      level = self.next(level, c.index(action_so_far))
    return (level, best)


"""Makes and returns a cascade from the supplied component classes."""
def make_cascade(name, actions, component_class_list):
  ls = []
  for c in component_class_list:
    ls.append(c(actions))
  cascade = Cascade(ls, actions)
  cascade.set_name(name)
  return cascade


"""Flat cascade with only one top-level component."""
class Flat(Cascade):
  def __init__(self, actions):
    components = [FlatComponent(actions)]
    super(Flat, self).__init__(components, actions)


"""Two-level cascade that first decides to shift or not, and if not, decides
using a flat action space on the exact action.
"""
class ShiftOrNotCascade(Cascade):
  def __init__(self, actions):
    components = [ShiftOrNotComponent(actions), NonShiftComponent(actions)]
    super(ShiftOrNotCascade, self).__init__(components, actions)
    self.connect(parent=0, index=ShiftOrNotComponent.NOT_SHIFT, child=1)


"""Three-level cascade that decides:
- Shift or not.
- If not, then whether to evoke a Propbank frame or do one of the other
  actions.
- if a Propbank evoke, then the specific evoke.
"""
class ShiftPropbankEvokeCascade(Cascade):
  def __init__(self, actions):
    components = [ShiftOrNotComponent(actions), \
                  ExceptShiftAndPropbankEvokeComponent(actions), \
                  PropbankEvokeComponent(actions)]
    super(ShiftPropbankEvokeCascade, self).__init__(components, actions)
    self.connect(0, ShiftOrNotComponent.NOT_SHIFT, 1)
    self.connect(1, components[1].pb_index, 2)


"""Prints softmax speedup headroom for various cascades."""
def speedup_headroom():
  import train_util as utils
  from corpora import Corpora
  path = "/home/grahul/sempar_ontonotes/"
  resources = utils.Resources()
  resources.load(path + "commons", path + "train.rec")

  actions = resources.spec.actions
  for cascade in [Flat, ShiftOrNotCascade, ShiftPropbankEvokeCascade]:
    cascade(actions).build(print_cost_estimate=True)

