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


# Represents a single transition.
class Action:
  SHIFT = 0
  STOP = 1
  EVOKE = 2
  REFER = 3
  CONNECT = 4
  ASSIGN = 5
  EMBED = 6
  ELABORATE = 7
  NUM_ACTIONS = 8

  def __init__(self, t=None):
    self.type = None
    self.length = None
    self.source = None
    self.target = None
    self.role = None
    self.label = None

    if t is not None:
      assert t < Action.NUM_ACTIONS
      assert t >= 0
      self.type = t


  # Converts the action to a tuple.
  def _totuple(self):
    return (self.type, self.length, self.source, self.target,
            self.role, self.label)


  # Methods to enable hashing of actions.
  def __hash__(self):
    return hash(self._totuple())


  def __eq__(self, other):
    return self._totuple() == other._totuple()


  # Copies non-None fields to 'other'.
  def copy_to(self, other):
    if self.type is not None: other.type = self.type
    if self.length is not None: other.length = self.length
    if self.label is not None: other.label = self.label
    if self.source is not None: other.source = self.source
    if self.target is not None: other.target = self.target
    if self.role is not None: other.role = self.role


  # Returns whether the action has type 't'.
  def typed(self, t):
    return self.type == t


  # Returns whether the action is a SHIFT.
  def is_shift(self):
    return self.type == Action.SHIFT


  # Returns whether the action is a STOP.
  def is_stop(self):
    return self.type == Action.STOP


  # Returns the string representation of the action.
  def __repr__(self):
    names = {
      Action.SHIFT: "SHIFT",
      Action.STOP: "STOP",
      Action.EVOKE: "EVOKE",
      Action.REFER: "REFER",
      Action.CONNECT: "CONNECT",
      Action.ASSIGN: "ASSIGN",
      Action.EMBED: "EMBED",
      Action.ELABORATE: "ELABORATE"
    }

    s = names[self.type]
    for k, v in sorted(self.__dict__.iteritems()):
      if v is not None and k != "type":
        if s != "":
          s = s + ", "
        s = s + k + ": " + str(v)
    return s


