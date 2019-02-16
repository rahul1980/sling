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


import sling
import sling.log as log


def load_kb(task):
  kb = sling.Store()
  if type(task) is str:
    kb.load(task)  # assume as filename
  else:
    kb.load(task.input("kb").name)
  log.info("Knowledge base read")
  kb.lockgc()
  kb.freeze()
  kb.unlockgc()
  log.info("Knowledge base frozen")
  return kb


