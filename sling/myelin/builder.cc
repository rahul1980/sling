// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sling/myelin/builder.h"

namespace sling {
namespace myelin {

Flow::Variable *Builder::Var(const string &name, Type type,
                             const Shape &shape) {
  return flow_->AddVariable(func_->name + "/" + name, type, shape);
}

Flow::Variable *Builder::Op(const string &op,
                            const std::vector<Flow::Variable *> &args) {
  string name = OpName(op);
  Variable *result = flow_->AddVariable(name + ":0", DT_INVALID, {0});
  flow_->AddOperation(func_, name, op, args, {result});
  return result;
}

void Builder::Op0(const string &op,
                  const std::vector<Flow::Variable *> &args) {
  string name = OpName(op);
  flow_->AddOperation(func_, name, op, args, {});
}

Flow::Variable *Builder::Constant(const void *data, Type type,
                                  const Shape &shape) {
  Variable *var = Var(OpName("const"), type, shape);
  var->size = TypeTraits::of(type).size() * shape.elements();
  char *buffer = flow_->AllocateMemory(var->size);
  var->data = buffer;
  memcpy(buffer, data, var->size);
  return var;
}

Flow::Variable *Builder::Instance(Function *func) {
  Variable *instance = Var(func->name, DT_RESOURCE, {});
  instance->ref = true;
  return instance;
}

Flow::Variable *Builder::Ref(Variable *instance, Variable *external) {
  Variable *ref = Op("Reference", {instance});
  ref->type = external->type;
  ref->shape = external->shape;
  ref->ref = true;
  ref->producer->SetAttr("var", external->name);
  return ref;
}

string Builder::OpName(const string &op) {
  string name = func_->name;
  name.push_back('/');
  name.append(op);
  int num = opnum_[op]++;
  if (num > 0) {
    name.push_back('_');
    name.append(std::to_string(num));
  }
  return name;
}

}  // namespace myelin
}  // namespace sling

