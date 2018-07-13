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

#include <math.h>

#include "sling/nlp/parser/parser.h"

#include "sling/frame/serialization.h"
#include "sling/myelin/cuda/cuda-runtime.h"
#include "sling/myelin/kernel/cuda.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/myelin/kernel/tensorflow.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/features.h"
#include "sling/nlp/document/lexicon.h"
#include "sling/string/strcat.h"

REGISTER_COMPONENT_REGISTRY("delegate runtime", sling::nlp::Delegate);

namespace sling {
namespace nlp {

static myelin::CUDARuntime cudart;

// Delegate that assumes a softmax output.
class SoftmaxDelegate : public Delegate {
 public:
  void Initialize(const Cascade *cascade,
                  const myelin::Cell *cell,
                  const Frame &spec) override {
    input_ = cell->GetParameter(cell->name() + "/input");
    output_ = cell->GetParameter(cell->name() + "/output");

    // Read delegate's action table.
    Store *store = spec.store();
    Array actions(store, spec.GetHandle("actions"));
    CHECK(actions.valid()) << name();
    for (int i = 0; i < actions.length(); ++i) {
      ParserAction action;
      Frame frame(store, actions.get(i));
      action.type =
        static_cast<ParserAction::Type>(frame.GetInt("/table/action/type"));
      if (frame.Has("/table/action/delegate")) {
        action.delegate = frame.GetInt("/table/action/delegate");
      }
      if (frame.Has("/table/action/length")) {
        action.length = frame.GetInt("/table/action/length");
      }
      if (frame.Has("/table/action/source")) {
        action.source = frame.GetInt("/table/action/source");
      }
      if (frame.Has("/table/action/target")) {
        action.target = frame.GetInt("/table/action/target");
      }
      if (frame.Has("/table/action/label")) {
        action.label = frame.GetHandle("/table/action/label");
      }
      if (frame.Has("/table/action/role")) {
        action.role = frame.GetHandle("/table/action/role");
      }
      actions_.push_back(action);
    }
    LOG(INFO) << name() << " has " << actions_.size() << " actions";
  }

  void Compute(
    myelin::Instance *instance, ParserAction *action) const override {
    int best_index = *instance->Get<int>(output_);

    // NOTE: A more general and slightly more expensive approach would be
    // to call another virtual method here:
    //   Overlay(actions_[best_index], action);
    // Right now we overwrite the under-construction action with the output.
    *action = actions_[best_index];
  }

 private:
  // Location of the delegate output (argmax of the softmax layer).
  myelin::Tensor *output_ = nullptr;

  // Action table for the delegate.
  std::vector<ParserAction> actions_;
};

REGISTER_DELEGATE_RUNTIME("SoftmaxDelegate", SoftmaxDelegate);

Cascade::~Cascade() {
  for (auto *d : delegates_) delete d;
}

void Cascade::Initialize(const myelin::Network &network, const Frame &spec) {
  Store *store = spec.store();
  Array delegates(store, spec.GetHandle("delegates"));
  CHECK(delegates.valid());

  // Create delegates.
  std::vector<Frame> delegate_specs;
  for (int i = 0; i < delegates.length(); ++i) {
    Frame frame(store, delegates.get(i));
    string runtime = frame.GetText("runtime").str();

    Delegate *d = Delegate::Create(runtime);
    d->set_cell(network.GetCell(StrCat("delegate", i)));
    d->set_name(frame.GetText("name").str());
    d->set_runtime(runtime);

    delegates_.push_back(d);
    delegate_specs.push_back(frame);
  }

  // Initialize delegates. Delegates can choose to access other delegates in
  // the cascade at this point.
  int i = 0;
  for (auto *d : delegates_) {
    d->Initialize(this, d->cell(), delegate_specs[i++]);
  }
}

CascadeInstance *Cascade::CreateInstance() const {
  return new CascadeInstance(this);
}

CascadeInstance::CascadeInstance(const Cascade *cascade)
    : cascade_(cascade),
      shift_(ParserAction::SHIFT),
      stop_(ParserAction::STOP) {
  for (auto *d : cascade->delegates_) {
    instances_.push_back(new myelin::Instance(d->cell()));  
  }
}

CascadeInstance::~CascadeInstance() {
  for (auto *i : instances_) delete i;
}

void CascadeInstance::Compute(myelin::Channel *activations,
                              int step,
                              ParserState *state,
                              ParserAction *output) {
  int current = 0;
  while (true) {
    // Execute the current delegate.
    Delegate *delegate = cascade_->delegates_[current];
    myelin::Instance *instance = instances_[current];
    instance->Clear();
    instance->Set(delegate->input(), activations, step);
    instance->Compute();
    delegate->Compute(instance, output);

    // If there is a cascade down the chain then follow it.
    // To avoid potential infinite loops, cascades to delegates
    // up in the chain are disallowed.
    bool is_cascade = output->type == ParserAction::CASCADE;
    if (is_cascade && (output->delegate > current)) {
      current = output->delegate;
      continue;
    }

    // If we have an applicable action then we are done with the cascade.
    if (!is_cascade && state->CanApply(*output)) return;

    // Return a fallback action.
    *output = (state->current() < state->end()) ? shift_ : stop_;
    return;
  }
}

void Parser::EnableGPU() {
  if (myelin::CUDA::Supported()) {
    // Initialize CUDA runtime for Myelin.
    if (!cudart.connected()) {
      cudart.Connect();
    }

    // Always use fast fallback when running on GPU.
    use_gpu_ = true;
  }
}

void Parser::Load(Store *store, const string &model) {
  // Register kernels for implementing parser ops.
  RegisterTensorflowLibrary(&library_);
  RegisterDragnnLibrary(&library_);
  if (use_gpu_) RegisterCUDALibrary(&library_);

  // Load and analyze parser flow file.
  myelin::Flow flow;
  CHECK(flow.Load(model));

  // Analyze parser flow file.
  flow.Analyze(library_);

  // Compile parser flow.
  if (use_gpu_) network_.set_runtime(&cudart);
  CHECK(network_.Compile(flow, library_));

  // Initialize lexical encoder.
  encoder_.Initialize(network_);
  encoder_.LoadLexicon(&flow);

  // Initialize feed-forward trunk.
  InitFF("ff_trunk", &ff_);

  // Load commons and action stores.
  myelin::Flow::Blob *commons = flow.DataBlock("commons");
  if (commons != nullptr) {
    StringDecoder decoder(store, commons->data, commons->size);
    decoder.DecodeAll();
  }
  myelin::Flow::Blob *actions = flow.DataBlock("actions");
  if (actions != nullptr) {
    StringDecoder decoder(store, actions->data, actions->size);
    decoder.DecodeAll();
  }

  // Read the cascade specification and implementation from the flow.
  myelin::Flow::Blob *cascade = flow.DataBlock("cascade");
  CHECK(cascade != nullptr);
  {
    StringDecoder decoder(store, cascade->data, cascade->size);
    decoder.DecodeAll();
    Frame spec(store, "cascade");
    CHECK(spec.valid());
    cascade_.Initialize(network_, spec);
  }

  // Initialize action table.
  store_ = store;
  actions_.Init(store);
  frame_limit_ = actions_.frame_limit();
  roles_.Init(actions_);
}

void Parser::InitFF(const string &name, FF *ff) {
  // Get cell.
  ff->cell = GetCell(name);

  // Get feature inputs.
  ff->lr_focus_feature = GetParam(name + "/lr", true);
  ff->rl_focus_feature = GetParam(name + "/rl", true);
  ff->lr_attention_feature = GetParam(name + "/frame-end-lr", true);
  ff->rl_attention_feature = GetParam(name + "/frame-end-rl", true);
  ff->frame_create_feature = GetParam(name + "/frame-creation-steps", true);
  ff->frame_focus_feature = GetParam(name + "/frame-focus-steps", true);
  ff->history_feature = GetParam(name + "/history", true);
  ff->out_roles_feature = GetParam(name + "/out-roles", true);
  ff->in_roles_feature = GetParam(name + "/in-roles", true);
  ff->unlabeled_roles_feature = GetParam(name + "/unlabeled-roles", true);
  ff->labeled_roles_feature = GetParam(name + "/labeled-roles", true);

  // Get feature sizes.
  std::vector<myelin::Tensor *> attention_features {
    ff->lr_attention_feature,
    ff->rl_attention_feature,
    ff->frame_create_feature,
    ff->frame_focus_feature,
  };
  for (auto *f : attention_features) {
    if (!f) continue;
    if (f->elements() > ff->attention_depth) {
      ff->attention_depth = f->elements();
    }
  }
  for (auto *f : attention_features) {
    if (!f) continue;
    CHECK_EQ(ff->attention_depth, f->elements());
  }
  if (ff->history_feature != nullptr) {
    ff->history_size = ff->history_feature->elements();
  }
  if (ff->out_roles_feature != nullptr) {
    ff->out_roles_size = ff->out_roles_feature->elements();
  }
  if (ff->in_roles_feature != nullptr) {
    ff->in_roles_size = ff->in_roles_feature->elements();
  }
  if (ff->unlabeled_roles_feature != nullptr) {
    ff->unlabeled_roles_size = ff->unlabeled_roles_feature->elements();
  }
  if (ff->labeled_roles_feature != nullptr) {
    ff->labeled_roles_size = ff->labeled_roles_feature->elements();
  }

  // Get links.
  ff->lr_lstm = GetParam(name + "/link/lr_lstm");
  ff->rl_lstm = GetParam(name + "/link/rl_lstm");
  ff->steps = GetParam(name + "/steps");

  ff->hidden = GetParam(name + "/hidden");
}

void Parser::Parse(Document *document) const {
  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Initialize parser model instance data.
    ParserInstance data(this, document, s.begin(), s.end());
    LexicalEncoderInstance &encoder = data.encoder_;

    // Run the lexical encoder.
    auto bilstm = encoder.Compute(*document, s.begin(), s.end());

    // Run FF to predict transitions.
    ParserState &state = data.state_;
    bool done = false;
    int steps_since_shift = 0;
    int step = 0;
    while (!done) {
      // Allocate space for next step.
      data.ff_step_.push();

      // Attach instance to recurrent layers.
      data.ff_.Clear();
      data.AttachFF(step, bilstm);

      // Extract features.
      data.ExtractFeaturesFF(step);

      // Compute FF hidden layer.
      data.ff_.Compute();

      // Apply the cascade.
      ParserAction action;
      data.cascade_->Compute(&data.ff_step_, step, &state, &action);
      state.Apply(action);

      // Update state.
      switch (action.type) {
        case ParserAction::SHIFT:
          steps_since_shift = 0;
          break;

        case ParserAction::STOP:
          done = true;
          break;

        case ParserAction::CASCADE:
          LOG(FATAL) << "CASCADE action should not reach ParserState.";
          break;

        case ParserAction::EVOKE:
        case ParserAction::REFER:
        case ParserAction::CONNECT:
        case ParserAction::ASSIGN:
        case ParserAction::EMBED:
        case ParserAction::ELABORATE:
          steps_since_shift++;
          if (state.AttentionSize() > 0) {
            int focus = state.Attention(0);
            if (data.create_step_.size() < focus + 1) {
              data.create_step_.resize(focus + 1);
              data.create_step_[focus] = step;
            }
            if (data.focus_step_.size() < focus + 1) {
              data.focus_step_.resize(focus + 1);
            }
            data.focus_step_[focus] = step;
          }
      }

      // Next step.
      step += 1;
    }

    // Add frames for sentence to the document.
    state.AddParseToDocument(document);
  }
}

myelin::Cell *Parser::GetCell(const string &name) {
  myelin::Cell *cell = network_.LookupCell(name);
  if (cell == nullptr) {
    LOG(FATAL) << "Unknown parser cell: " << name;
  }
  return cell;
}

myelin::Tensor *Parser::GetParam(const string &name, bool optional) {
  myelin::Tensor *param = network_.LookupParameter(name);
  if (param == nullptr && !optional) {
    LOG(FATAL) << "Unknown parser parameter: " << name;
  }
  return param;
}

ParserInstance::ParserInstance(const Parser *parser, Document *document,
                               int begin, int end)
    : parser_(parser),
      encoder_(parser->encoder()),
      state_(document->store(), begin, end),
      ff_(parser->ff_.cell),
      ff_step_(parser->ff_.hidden) {
  // Reserve two transitions per token.
  int length = end - begin;
  ff_step_.reserve(length * 2);

  // Setup cascade instance.
  cascade_ = parser->cascade_.CreateInstance();
}

void ParserInstance::AttachFF(int output, const myelin::BiChannel &bilstm) {
  ff_.Set(parser_->ff_.lr_lstm, bilstm.lr);
  ff_.Set(parser_->ff_.rl_lstm, bilstm.rl);
  ff_.Set(parser_->ff_.steps, &ff_step_);
  ff_.Set(parser_->ff_.hidden, &ff_step_, output);
}

void ParserInstance::ExtractFeaturesFF(int step) {
  // Extract LSTM focus features.
  const Parser::FF &ff = parser_->ff_;
  int current = state_.current() - state_.begin();
  if (state_.current() == state_.end()) current = -1;
  int *lr_focus = GetFF(ff.lr_focus_feature);
  int *rl_focus = GetFF(ff.rl_focus_feature);
  if (lr_focus != nullptr) *lr_focus = current;
  if (rl_focus != nullptr) *rl_focus = current;

  // Extract frame attention, create, and focus features.
  if (ff.attention_depth > 0) {
    int *lr = GetFF(ff.lr_attention_feature);
    int *rl = GetFF(ff.rl_attention_feature);
    int *create = GetFF(ff.frame_create_feature);
    int *focus = GetFF(ff.frame_focus_feature);
    for (int d = 0; d < ff.attention_depth; ++d) {
      int att = -1;
      int created = -1;
      int focused = -1;
      if (d < state_.AttentionSize()) {
        // Get frame from attention buffer.
        int frame = state_.Attention(d);

        // Get end token for phrase that evoked frame.
        att = state_.FrameEvokeEnd(frame);
        if (att != -1) att -= state_.begin() + 1;

        // Get the step numbers that created and focused the frame.
        created = create_step_[frame];
        focused = focus_step_[frame];
      }
      if (lr != nullptr) lr[d] = att;
      if (rl != nullptr) rl[d] = att;
      if (create != nullptr) create[d] = created;
      if (focus != nullptr) focus[d] = focused;
    }
  }

  // Extract history feature.
  int *history = GetFF(ff.history_feature);
  if (history != nullptr) {
    int h = 0;
    int s = step - 1;
    while (h < ff.history_size && s >= 0) history[h++] = s--;
    while (h < ff.history_size) history[h++] = -1;
  }

  // Extract role features.
  if (parser_->frame_limit_ > 0 && parser_->roles_.size() > 0) {
    // Construct role graph for center of attention.
    RoleGraph graph;
    graph.Compute(state_, parser_->frame_limit_, parser_->roles_);

    // Extract out roles.
    int *out = GetFF(ff.out_roles_feature);
    if (out != nullptr) {
      int *end = out + ff.out_roles_size;
      graph.out([&out, end](int f) {
        if (out < end) *out++ = f;
      });
      while (out < end) *out++ = -2;
    }

    // Extract in roles.
    int *in = GetFF(ff.in_roles_feature);
    if (in != nullptr) {
      int *end = in + ff.in_roles_size;
      graph.in([&in, end](int f) {
        if (in < end) *in++ = f;
      });
      while (in < end) *in++ = -2;
    }

    // Extract unlabeled roles.
    int *unlabeled = GetFF(ff.unlabeled_roles_feature);
    if (unlabeled != nullptr) {
      int *end = unlabeled + ff.unlabeled_roles_size;
      graph.unlabeled([&unlabeled, end](int f) {
        if (unlabeled < end) *unlabeled++ = f;
      });
      while (unlabeled < end) *unlabeled++ = -2;
    }

    // Extract labeled roles.
    int *labeled = GetFF(ff.labeled_roles_feature);
    if (labeled != nullptr) {
      int *end = labeled + ff.labeled_roles_size;
      graph.labeled([&labeled, end](int f) {
        if (labeled < end) *labeled++ = f;
      });
      while (labeled < end) *labeled++ = -2;
    }
  }
}

}  // namespace nlp
}  // namespace sling

