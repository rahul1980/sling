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
from collections import defaultdict

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import parse_qs
import cgi
import SocketServer

#kb = sling.Store()
#kb.load("local/data/e/wiki/kb.sling")
#kb.lockgc()
#kb.freeze()
#kb.unlockgc()
#print "KB loaded"

filename = "local/data/e/wikicat/filtered-parses.rec"
reader = sling.RecordReader(filename)
name_to_qid = {}
signature_to_category = defaultdict(list)
for key, value in reader:
  store = sling.Store()
  frame = store.parse(value)
  name_to_qid[frame.name] = key
  for parse in frame("parse"):
    signature = ' '.join([x for x in parse.signature])
    signature_to_category[signature].append((key, frame.name, parse.score))

for _, value in signature_to_category.iteritems():
  value.sort(key=lambda x: -x[2])

db = sling.RecordDatabase(filename)


class Browser(BaseHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()

  def _begin(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += ">"
    self.wfile.write(s)

  def _begin_end(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += "/>"
    self.wfile.write(s)

  def _end(self, tag):
    if type(tag) is list:
      for n in tag:
        self.wfile.write("</" + n + ">")
    else:
      self.wfile.write("</" + tag + ">")

  def _text(self, text):
    self.wfile.write(str(text))

  def _color_text(self, text, color=None, hover=None):
    style = None
    if color is not None:
      style = 'color:' + color
    self._tag("span", text, style=style, title=hover)

  def _tag(self, tag, text, **kwargs):
    self._begin(tag, **kwargs)
    if text is not None:
      self.wfile.write(str(text))
    self._end(tag)

  def _br(self):
    self._begin_end("br")

  def write_form(self, form=None):
    self._begin("html")
    self._begin("head")
    self._begin("script", language="javascript")
    self._text('\nfunction new_qid(qid) {')
    self._text('\n  document.getElementById("category_qid").value = qid;')
    self._text('\n  document.getElementById("main_form").submit();')
    self._text('\n}')
    self._end(["script", "head"])
    self._text('\n')
    self._begin("body")
    self._begin("form", id="main_form", method="POST", action="")
    self._text(" Loaded %d categories with %d signatures" % \
      (len(name_to_qid), len(signature_to_category)))
    self._br()
    self._text(" Enter category QID or name: ")

    value = form.getvalue("category") if form is not None else None
    self._begin_end("input", id="category_qid", name="category", \
      type="text", size=50, value=value)
    self._br()

    for checkbox in ["show_span_qid", "show_parse_scores", \
      "show_parse_rewards", "show_span_scores", "show_span_rewards", \
      "show_matching_categories"]:
      name = checkbox.replace("_", " ").capitalize()
      self._text(" " + name + ": ")
      value = form.getvalue(checkbox) if form is not None else None
      self._begin_end("input", name=checkbox, type="checkbox", checked=value)
      self._br()
    self._begin_end("input", type="submit")
    self._end(["form", "body", "html"])

  def do_HEAD(self):
    self._set_headers()

  def do_GET(self):
    self._set_headers()
    if self.path == "/":
      print self.headers
      if 'content-length' in self.headers:
        print parse_qs(self.rfile.read(int(self.headers['content-length'])))
      self.write_form()


  def do_POST(self):
    print "path", self.path
    self._set_headers()
    form = cgi.FieldStorage(
        fp=self.rfile,
        headers=self.headers,
        environ={'REQUEST_METHOD': 'POST'}
    )
    self.write_form(form)
    category = form.getvalue("category")
    if category in name_to_qid:
      category = name_to_qid[category]

    print "Looking for", category
    value = db.lookup(category)
    if value is None:
      self.wfile.write("<span style='color:red'>Unknown category</span>")
    else:
      self.write_parses(category, value, form)


  def write_parses(self, qid, value, form):
    def is_on(name):
      return form.getvalue(name) == "on"

    # Various options.
    show_span_qid = is_on("show_span_qid")
    show_parse_scores = is_on("show_parse_scores")
    show_parse_rewards = is_on("show_parse_rewards")
    show_span_scores = is_on("show_span_scores")
    show_span_rewards = is_on("show_span_rewards")
    show_matching_categories = is_on("show_matching_categories")

    store = sling.Store()
    frame = store.parse(value)
    document = sling.Document(frame=frame.document)

    num = len([p for p in frame("parse")])
    self._tag("div", "<b>" + frame.name + "</b>: " + str(num) + " parses")
    self._br()

    self._begin("table", border=1, cellpadding=10)
    self._begin("thead")
    for token in document.tokens:
      self._tag("th", token.word)

    need_separator = show_parse_scores or show_parse_rewards \
      or show_matching_categories

    if need_separator: self._tag("th", "", style="background-color:gray")
    if show_parse_scores: self._tag("th", "Scores")
    if show_parse_rewards: self._tag("th", "Rewards")
    if show_matching_categories: self._tag("th", "Similar Categories")
    self._end("thead")

    for parse in frame("parse"):
      self._begin("tr")
      prev_span_end = -1
      for span in parse.spans:
        for index in xrange(prev_span_end + 1, span.begin):
          self._tag("td", "&nbsp;")

        self._begin("td", colspan=span.end-span.begin, align='middle')
        text = span.signature
        if show_span_qid:
          text += " (" + str(span.qid) + ")"
        title = '.'.join([str(p) for p in span.pids]) + ' = ' + str(span.qid)
        if "name" in span.qid:
          title += " (" + span.qid[name] + ")"
        self._tag("span", text, title=title)

        if show_span_scores and "prior" in span:
          self._br()
          self._text("%s = %0.4f" % ("prior", span.prior))

        if show_span_rewards:
          if "reward" not in span:
            self._br()
            self._color_text("[No rewards]", "red")
          for reward in span("reward"):
            self._br()
            self._text("%s = %0.4f" % (reward.name, reward.value))

        self._end("td")
        prev_span_end = span.end - 1
      for index in xrange(prev_span_end + 1, len(document.tokens)):
        self._tag("td", "&nbsp;")

      if need_separator:
        self._tag("td", "", style="background-color:gray")

      if show_parse_scores:
        self._begin("td")
        for score_type in ["prior", "member_score", "cover"]:
          if score_type in parse:
            self._text("%s = %0.4f" % (score_type, parse[score_type]))
            self._br()
        if "score" in parse:
          self._color_text("Overall = %0.4f" % parse.score, "blue")
        self._end("td")

      if show_parse_rewards:
        self._begin("td")
        if "reward" not in parse:
          self._br()
          self._color_text("[No rewards]", "red")
        for reward in parse("reward"):
          self._text("%s = %0.4f" % (reward.name, reward.value))
          self._br()
        self._end("td")

      if show_matching_categories:
        self._begin("td")
        limit = 5
        sig = ' '.join([x for x in parse.signature])
        seen = set()
        for (other_qid, other_name, score) in signature_to_category[sig]:
          if len(seen) >= limit:
            break
          if other_qid != qid and other_qid not in seen:
            seen.add(other_qid)
            self._text(other_name)
            self._tag("a", " (= " + other_qid + ")", \
              onclick='new_qid("' + other_qid + '"); return false;', href='')
            self._text(" (%0.4f)" % score)
            self._br()
        self._end("td")


      self._end("tr")
    self._end("table")


def run(port=8001):
  server_address = ('', port)
  httpd = HTTPServer(server_address, Browser)
  print 'Starting HTTP Server on port', port
  httpd.serve_forever()

if __name__ == "__main__":
  run()

