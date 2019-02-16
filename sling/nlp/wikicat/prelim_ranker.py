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

# Task for outputting a preliminary ranking of parses, keeping only the top-k,
# and computing a signature for each surviving parse.
# A signature replaces each (PID, QID) span in a parse by $PID=$label(QID),
# where 'label' is derived using a custom taxonomy.

import math
import sling
from sling.task.workflow import register_task
from util import load_kb

class PrelimCategoryParseRanker:
  def init(self, task):
    self.kb = load_kb(task)
    self.extractor = sling.api.FactExtractor(self.kb)
    self.h_name = self.kb['name']

    # Custom type ordering for building a taxonomy.
    taxonomy_types = [
        'Q215627',     # person 25K
        'Q95074',      # fictional character 303
        'Q729',        # animal 25
        'Q4164871',    # position 17K
        'Q12737077',   # occupation 96K
        'Q216353',     # title 1.4K
        'Q618779',     # award 2.5K
        'Q31629',      # type of sport
        'Q27020041',   # sports season 2.5K
        'Q4438121',    # sports organization 35K
        'Q215380',     # band 8K
        'Q2385804',    # educational institution 25K
        'Q783794',     # company 20K
        'Q41710',      # ethnic group  (NEW TYPE) 11K
        'Q6256',       # country  (NEW TYPE) 194K
        'Q17334923',   # location
        'Q43229',      # organization
        'Q431289',     # brand
        'Q571',        # book 485
        'Q732577',     # publication 342
        'Q11424',      # film 361
        'Q15416',      # television program 1059
        'Q12136',      # disease 483
        'Q1931388',    # cause of death
        'Q16521',      # taxon 2.2K
        'Q5058355',    # cellular component 4
        'Q7187',       # gene 1
        'Q11173',      # chemical compound 45
        'Q811430',     # construction 108
        'Q618123',     # geographical object 1058
        'Q1656682',    # event 14K
        'Q101352',     # family name 2K
        'Q202444',     # given name 1.7K
        'Q577',        # year 84K
        'Q186081',     # time interval
        'Q11563',      # number
        'Q17376908',   # languoid
        'Q1047113',    # specialty  (REORDERED) 5.2K
        'Q968159',     # art movement (NEW TYPE) 772
        'Q483394',     # genre  (REORDERED) 20K
        'Q47574',      # unit of measurement (REPLACES unit) 38
        'Q3695082',    # sign 327
        'Q2996394',    # biological process 993
        'Q11410',      # game 523
        'Q7397',       # software 532
        'Q838948',     # work of art 126
        'Q47461344',   # written work 108
        'Q28877',      # goods
        'Q15401930',   # product
        'Q121769',     # reference
        'Q1190554'     # occurrence
    ]
    self.taxonomy = self.extractor.taxonomy(taxonomy_types)


  # Computes various kinds of scores for each parse and adds them as slots to
  # the parse frame. Returns a list of (parse, score) pairs.
  def score(self, category):
    parses = [p for p in category("parse")]
    output = []

    # Collect qid -> span boundaries across all parses.
    # This will be used to determine subsumption.
    qid_to_spans = {}
    for parse in parses:
      for span in parse.spans:
        if span.qid not in qid_to_spans:
          qid_to_spans[span.qid] = set()
        qid_to_spans[span.qid].add((span.begin, span.end))

    num_members = 1.0 * len(category.members)
    for parse in parses:
      prior = 1.0            # product of priors of all spans' QIDs
      member_score = 1.0     # product of proportions of members covered
      cover = 0              # number of title tokens covered
      subsumed = False       # whether a QID also belongs to a bigger span
      for span in parse.spans:
        for s in qid_to_spans[span.qid]:
          if s[1] - s[0] > span.end - span.begin and \
              s[1] >= span.end and s[0] <= span.begin:
            subsumed = True
            break
        prior *= span.prior
        member_score *= span.count / num_members
        cover += span.end - span.begin

      # Normalize all scores so that they can be compared across parses.
      prior = math.pow(prior, 1.0 / len(spans))
      member_score = math.pow(member_score, 1.0 / len(spans))
      cover /= 1.0 * len(category.document.tokens)

      overall_score = 0.0
      if not subsumed:
        overall_score = prior * member_score * cover

      # Adds various scores to the parse.
      parse["subsumed"] = subsumed
      parse["prior"] = prior
      parse["member_score"] = member_score
      parse["cover"] = cover
      parse["score"] = overall_score
      output.append((parse, overall_score))
    return output


  # Returns a signature for 'parse'. Spans in the parse are reported as
  # $PID=$label(QID), and tokens not covered by any span are reported as-is.
  def signature(self, document, parse, coarse=False):
    tokens = []              # tokens in the full signature
    span_signature = {}      # span -> span's signature
    start = 0
    for span in parse.spans:
      while start < span.begin:
        tokens.append(document.tokens[start].word)
        start += 1
      label = self.taxonomy.classify(span.qid)
      if label is None:
        label = span.qid
      elif self.h_name in label:
        label = label[self.h_name]
      if coarse:
        word = str(label)
      else:
        pids = '.'.join([pid.name for pid in span.pids])
        word = '$' + pids + '=$' + str(label)
      word = word.replace(' ', '_')
      span_signature[span] = word
      tokens.append(word)
      start = span.end
    while start < len(document.tokens):
      tokens.append(document.tokens[start].word)
      start += 1
    return tokens, span_signature


  # Generates preliminary ranking of parses and their signatures.
  def run(self, task):
    self.init(task)

    max_parses = int(task.param("max_parses"))
    reader = sling.RecordReader(task.input("input").name)
    writer = sling.RecordWriter(task.output("output").name)
    for index, (key, value) in enumerate(reader):
      store = sling.Store(self.kb)
      category = store.parse(value)
      document = sling.Document(category.document)

      # Score each parse.
      parse_with_score = self.score(category)

      # Keep only the top-k parses.
      ranked_parses = sorted(parse_with_score, key=lambda x:-x[1])
      if len(ranked_parses) > max_parses:
        dropped = len(ranked_parses) - max_parses
        ranked_parses = ranked_parses[0:max_parses]
        task.increment("parses-dropped", dropped)
        task.increment("categories-with-too-many-parses")

      # Compute signature for each parse and store it in the parse.
      for parse, _ in ranked_parses:
        tokens, span_signature = self.signature(document, parse)
        parse["signature"] = tokens
        for span in parse.spans:
          if span in span_signature:
            span["signature"] = span_signature[span]

        # Also compute the coarse signature.
        tokens, span_signature = self.signature(document, parse, coarse=True)
        parse["coarse_signature"] = tokens
        for span in parse.spans:
          if span in span_signature:
            span["coarse_signature"] = span_signature[span]

      # Replace the current set of parses with the ranked list.
      del category["parse"]
      for parse, _ in ranked_parses:
        category.append("parse", parse)
      task.increment("parses-kept", len(ranked_parses))
      writer.write(key, category.data(binary=True))
    reader.close()
    writer.close()


register_task("prelim-category-parse-ranker", PrelimCategoryParseRanker)
