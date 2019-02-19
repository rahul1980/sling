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


"""
Browser for viewing category parses.
"""

import cgi
import sling
import sling.flags as flags
import sling.log as log
import SocketServer

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from fact_matcher import FactMatchType

"""
The browser is implemented as a subclass of BaseHTTPRequestHandler, which is
constructed to handle the incoming HTTP request and deleted the request is
processed. Therefore we need to keep any persistent state (e.g. RecordDatabase)
outside the handler.
"""

class BrowserGlobals:
  def read(self, parses_filename):
    reader = sling.RecordReader(parses_filename)
    self.category_name_to_qid = {}
    self.full_signature_to_category = defaultdict(list)
    self.coarse_signature_to_category = defaultdict(list)
    for key, value in reader:
      store = sling.Store()
      frame = store.parse(value)
      self.category_name_to_qid[frame.name] = key
      for parse in frame("parse"):
        full_signature = ' '.join([x for x in parse.signature])
        self.full_signature_to_category[full_signature].append(
            (key, frame.name, parse.score))
        coarse_signature = ' '.join([x for x in parse.coarse_signature])
        self.coarse_signature_to_category[coarse_signature].append(
            (key, frame.name, parse.score))

    # Arrange parses in descending order of scores.
    for _, value in self.full_signature_to_category.iteritems():
      value.sort(key=lambda x: -x[2])

    for _, value in self.coarse_signature_to_category.iteritems():
      value.sort(key=lambda x: -x[2])
    self.db = sling.RecordDatabase(parses_filename)


# This will be visible inside the browser, which is a BaseHTTPRequestHandler.
browser_globals = BrowserGlobals()


class Browser(BaseHTTPRequestHandler):
  # Sends default HTTP response headers.
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()


  # Various utility methods to write HTML response.
  #
  # Writes a beginning HTML tag. Tag attributes are taken from 'kwargs'.
  def _begin(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += ">"
    self.wfile.write(s)


  # Writes a beginning and ending HTML tag, e.g.
  # <input type='text' name='foo' />
  def _begin_end(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += "/>"
    self.wfile.write(s)


  # Writes ending HTML tag. If 'tag' is a list of HTML tag names,
  # then ending tags for all members of the list are generated in order.
  def _end(self, tag):
    if type(tag) is list:
      for n in tag:
        self.wfile.write("</" + n + ">")
    else:
      self.wfile.write("</" + tag + ">")


  # Writes the string version of 'text'.
  def _text(self, text):
    self.wfile.write(str(text))


  # Writes HTML to render 'text' in 'color', with a tooltip of 'hover'.
  def _color_text(self, text, color=None, hover=None):
    style = None
    if color is not None:
      style = 'color:' + color
    self._tag("span", text, style=style, title=hover)


  # Writes a beginning tag, some text, and end of tag.
  # Tag attributes are taken from 'kwargs'.
  def _tag(self, tag, text, **kwargs):
    self._begin(tag, **kwargs)
    if text is not None:
      self.wfile.write(str(text))
    self._end(tag)


  # Writes an HTML linebreak.
  def _br(self):
    self._begin_end("br")


  # Writes CSS styles to the response.
  def write_styles(self):
    styles = '''
      <style type="text/css">
        table.span_fact_match {
          font-size:11px;
          color:#333333;
          border-width: 1px;
          background-color: #f7f2f2;
          text-align: center;
          border-collapse: collapse;
        }
      </style>
      '''
    self.wfile.write(styles)


  # Writes the form at the top of the page.
  # If this HTTP request is the result of a form submission,
  # then the submitted form is present in 'form', whose
  # contents are duplicated in the form written by this method.
  def write_form(self, form=None):
    self._begin("html")
    self._begin("head")
    self.write_styles()
    self._begin("script", language="javascript")
    self._text('\nfunction new_qid(qid) {')
    self._text('\n  document.getElementById("main_input").value = qid;')
    self._text('\n  document.getElementById("main_form").submit();')
    self._text('\n}')
    self._end(["script", "head"])
    self._text('\n')
    self._begin("body")
    self._begin("form", id="main_form", method="POST", action="")
    self._text(" Loaded %d categories with %d full and %d coarse signatures" % \
      (len(browser_globals.category_name_to_qid),
       len(browser_globals.full_signature_to_category),
       len(browser_globals.coarse_signature_to_category)))
    self._br()

    # Generate the main input box.
    self._text(" Enter category QID/name or full/coarse signature: ")
    value = form.getvalue("main_input") if form is not None else None
    self._begin_end("input", id="main_input", name="main_input", \
      type="text", size=50, value=value)
    self._br()

    # Generate checkboxes for all the options.
    for checkbox in [
        "show_span_qid", "show_parse_scores", "show_span_scores",
        "show_fact_match_statistics", "show_span_level_fact_match_statistics",
        "show_similar_categories"]:
      name = checkbox.replace("_", " ").capitalize()
      self._text(" " + name + ": ")
      value = form.getvalue(checkbox) if form is not None else None
      if form is None and checkbox == "show_fact_match_statistics":
        value = "on"
      self._begin_end("input", name=checkbox, type="checkbox", checked=value)
      self._br()

    # Submit button.
    self._begin_end("input", type="submit")
    self._end(["form", "body", "html"])


  # Overridden method for generating the head of the response.
  def do_HEAD(self):
    self._set_headers()


  # Overridden method for responding to GET requests. This is called at the very
  # start when the browser is loaded, and also to get the page's thumbnail icon.
  def do_GET(self):
    self._set_headers()
    if self.path == "/":
      # For the first landing on the page, just generate the empty form.
      self.write_form()


  # Overridden method for responding to POST requests. This is the main method,
  # which is called whenever the form is submitted.
  def do_POST(self):
    self._set_headers()

    # Parse the form fields.
    form = cgi.FieldStorage(
        fp=self.rfile,
        headers=self.headers,
        environ={'REQUEST_METHOD': 'POST'}
    )

    # Mirror the filled out form in the response.
    self.write_form(form)

    # See if the input is a category name or qid.
    main_input = form.getvalue("main_input")
    if main_input in browser_globals.category_name_to_qid:
      main_input = browser_globals.category_name_to_qid[main_input]

    if main_input[0] == 'Q' and main_input[1:].isdigit():
      self.handle_category(main_input, form)
    elif main_input in browser_globals.coarse_signature_to_category:
      self.handle_coarse_signature(main_input, form)
    elif main_input in browser_globals.full_signature_to_category:
      self.handle_full_signature(main_input, form)
    else:
      self.bad_input("Can't handle input: %s" % main_input)


  # Generates an error message for bad inputs.
  def bad_input(self, message):
    self._color_text(message, color="red")


  # Handler for category inputs.
  def handle_category(self, qid, form):
    value = browser_globals.db.lookup(qid)
    if value is None:
      self.bad_input("Couldn't find category: %s" % qid)
    else:
      self.write_parses(qid, value, form)


  # Writes all the parses for a given category qid.
  def write_parses(self, qid, value, form):
    def is_on(name):
      return form.getvalue(name) == "on"

    def add_column(label):
      self._tag("th", str(label))

    def separate(head_row=True):
      tag = "th" if head_row else "td"
      self._tag(tag, "", style="background-color:gray")

    def cell(contents):
      self._tag("td", str(contents))

    def empty_cell():
      self._tag("td", "&nbsp;")

    # Various options.
    show_span_qid = is_on("show_span_qid")
    show_parse_scores = is_on("show_parse_scores")
    show_span_scores = is_on("show_span_scores")
    show_fact_matches = is_on("show_fact_match_statistics")
    show_span_fact_matches = is_on("show_span_level_fact_match_statistics")
    show_similar_categories = is_on("show_similar_categories")

    store = sling.Store()
    frame = store.parse(value)
    document = sling.Document(frame=frame.document)

    num = len([p for p in frame("parse")])
    self._tag("div", "<b>%s = %s</b>: %d members, %d parses" % \
              (qid, frame.name, len(frame.members), num))
    self._br()

    # Write the parses in a tabular format.
    self._begin("table", border=1, cellpadding=10)
    self._begin("thead")
    for token in document.tokens:
      add_column(token.word)

    need_separator =\
      show_parse_scores or show_fact_matches or show_similar_categories

    fact_match_types = [t for t in FactMatchType]
    if need_separator:
      separate()
    if show_parse_scores:
      add_column("Scores")
    if show_fact_matches:
      separate()
      for match_type in fact_match_types:
        add_column(match_type.name)
      separate()
    if show_similar_categories:
      add_column("Matching Categories")
    self._end("thead")

    # Each parse is written as one row.
    for parse in frame("parse"):
      total_match_type_count = defaultdict(int)
      self._begin("tr")
      prev_span_end = -1
      for span in parse.spans:
        for index in xrange(prev_span_end + 1, span.begin):
          empty_cell()

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

        if show_span_fact_matches:
          self._br()
          self._begin("table class='span_fact_match' border='1' cellpadding=2")
          self._begin("thead")
          for t in fact_match_types:
            self._tag("th", t.name)
          self._end("thead")

        if "fact_matches" in span:
          local_counts = {}
          for bucket in span.fact_matches.buckets:
            match_type = FactMatchType[bucket.match_type]
            local_counts[match_type] = bucket.count
            total_match_type_count[match_type] += bucket.count

        if show_span_fact_matches:
          self._begin("tr")
          for t in fact_match_types:
            count = "-" if t not in local_counts else local_counts[t]
            cell(count)
          self._end(["tr", "table"])

        self._end("td")
        prev_span_end = span.end - 1

      for index in xrange(prev_span_end + 1, len(document.tokens)):
        empty_cell()

      if need_separator:
        separate(head_row=False)

      if show_parse_scores:
        self._begin("td")
        for score_type in ["prior", "member_score", "cover"]:
          if score_type in parse:
            self._text("%s = %0.4f" % (score_type, parse[score_type]))
            self._br()
        if "score" in parse:
          self._color_text("Overall = %0.4f" % parse.score, "blue")
        self._end("td")

      if show_fact_matches:
        separate(head_row=False)
        for t in fact_match_types:
          count = "-"
          if t in total_match_type_count:
            count = total_match_type_count[t]
          cell(count)
        separate(head_row=False)

      if show_similar_categories:
        self._begin("td")
        limit = 5
        sig = ' '.join([x for x in parse.signature])
        seen = set()
        for (other_qid, other_name, score) in \
            browser_globals.full_signature_to_category[sig]:
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


def run(port, parses_rio):
  log.info('Reading parses from %s' % parses_rio)
  browser_globals.read(parses_rio)
  server_address = ('', port)
  httpd = HTTPServer(server_address, Browser)
  log.info('Starting HTTP Server on port %d' % port)
  httpd.serve_forever()


if __name__ == "__main__":
  flags.define("--port",
               help="port number for the HTTP server",
               default=8001,
               type=int,
               metavar="PORT")
  flags.define("--parses",
               help="Recordio of category parses",
               default="local/data/e/wikicat/parses-with-match-statistics.rec",
               type=str,
               metavar="FILE")
  flags.parse()
  run(flags.arg.port, flags.arg.parses)

