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

#include "myelin/compute.h"
#include "myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Reshape tensor while preserving the underlying data.
class Reshape : public Kernel {
 public:
  string Name() override { return "Reshape"; }
  string Operation() override { return "Reshape"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Removes dimensions of size 1 from the shape of a tensor while preserving the
// underlying data.
class Squeeze : public Kernel {
 public:
  string Name() override { return "Squeeze"; }
  string Operation() override { return "Squeeze"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Inserts a dimension of 1 into a tensor's shape while preserving the
// underlying data.
class ExpandDims : public Kernel {
 public:
  string Name() override { return "ExpandDims"; }
  string Operation() override { return "ExpandDims"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Divide "spatial" dimensions [1, ..., M] of the input, and interleaves these
// with the "batch" dimension (0).
class SpaceToBatch : public Kernel {
 public:
  string Name() override { return "SpaceToBatch"; }
  string Operation() override { return "SpaceToBatchND"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) {
      LOG(WARNING) << step->name() << " needs padding: "
                   << x->shape().ToString() << " -> " <<  y->shape().ToString();
    }
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Reshapes the "batch" dimension 0 into M + 1 dimensions, and interleaves these
// back into the spatial dimensions [1, ..., M].
class BatchToSpace : public Kernel {
 public:
  string Name() override { return "BatchToSpace"; }
  string Operation() override { return "BatchToSpaceND"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) {
      LOG(WARNING) << step->name() << " needs padding: "
                   << x->shape().ToString() << " -> " <<  y->shape().ToString();
    }
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Packs an array of rank-R tensors into one rank-(R+1) tensor.
class Pack : public Kernel {
 public:
  string Name() override { return "Pack"; }
  string Operation() override { return "Pack"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Unpacks an array of a rank-R tensor into rank-(R-1) tensors.
class Unpack : public Kernel {
 public:
  string Name() override { return "Unpack"; }
  string Operation() override { return "Unpack"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Output concatenation of input tensors.
class BasicConcat : public Kernel {
 public:
  string Name() override { return "BasicConcat"; }
  string Operation() override { return "ConcatV2"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() < 2 || step->outdegree() != 1) return false;

    // Only concatenation along a singular prefix supported.
    int n = step->GetAttr("N", step->indegree() - 1);
    if (step->indegree() < n + 1) return false;
    Tensor *axis = step->input(n);
    if (!axis->IsConstant()) return false;
    int dim = axis->value<int32>();
    if (step->output(0)->shape().outer(dim) != 1) return false;

    return true;
  }

  void Adjust(Step *step) override {
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);
    Register in = masm->rr().alloc();
    Register out = masm->rr().alloc();

    // Load output tensor.
    __ LoadTensorAddress(out, step->output(0));

    // Copy input tensors to output.
    int offset = 0;
    for (int i = 0; i < n; ++i) {
      int size = step->input(i)->size();
      if (size > 0 && size < 16) {
        __ LoadTensorAddress(in, step->input(i));
        int disp = offset;
        int left = size;
        while (left >= 8) {
          __ movq(acc, Operand(in, disp));
          __ movq(Operand(out, disp), acc);
          disp += 8;
          left -= 8;
        }
        while (left >= 4) {
          __ movl(acc, Operand(in, disp));
          __ movl(Operand(out, disp), acc);
          disp += 4;
          left -= 4;
        }
        while (left >= 2) {
          __ movw(acc, Operand(in, disp));
          __ movw(Operand(out, disp), acc);
          disp += 2;
          left -= 2;
        }
        while (left >= 1) {
          __ movb(acc, Operand(in, disp));
          __ movb(Operand(out, disp), acc);
          disp += 1;
          left -= 1;
        }
      } else {
        __ LoadTensorAddress(src, step->input(i));
        __ leaq(dst, Operand(out, offset));
        __ movq(cnt, Immediate(size));
        __ repmovsb();
      }
      offset += size;
    }
    CHECK_EQ(offset, step->output(0)->size());
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Register array kernels.
void RegisterArrayKernels(Library *library) {
  library->Register(new Reshape());
  library->Register(new Squeeze());
  library->Register(new ExpandDims());
  library->Register(new SpaceToBatch());
  library->Register(new BatchToSpace());
  library->Register(new Pack());
  library->Register(new Unpack());
  library->Register(new BasicConcat());
}

}  // namespace myelin
}  // namespace sling

