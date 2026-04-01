"""
search_gui.py
=============
Interactive web-based GUI for the Shakespeare Boolean Search Engine.

Place this file in the SAME directory as week_7.ipynb (and the ./books/ folder).
Run with:   python search_gui.py
Then open:  http://localhost:5000

Requires:  flask  (pip install flask)
All other dependencies are the same as the notebook (nltk, numpy, etc.).
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, time
from collections import defaultdict, Counter

# ── third-party ───────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, Response

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ── NLTK data (download once) ─────────────────────────────────────────────────
for pkg in ('punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'punkt_tab', 'omw-1.4'):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# =============================================================================
#  SECTION 1 — DATASET LOADING
# =============================================================================

BOOKS_DIR = './books'

raw_corpus  = {}   # doc_id (int) -> raw text
id_to_title = {}   # doc_id (int) -> title string

if os.path.isdir(BOOKS_DIR):
    for fname in sorted(os.listdir(BOOKS_DIR)):
        m = re.match(r'^(\d+)[\.\s]', fname)
        if m:
            doc_id = int(m.group(1))
            title  = os.path.splitext(fname)[0]
            title  = re.sub(r'^\d+[\.\s]+', '', title).strip()
            fpath  = os.path.join(BOOKS_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    raw_corpus[doc_id] = f.read()
                id_to_title[doc_id] = title
            except Exception as e:
                print(f'  [WARN] Could not read {fname}: {e}')

ALL_DOC_IDS = sorted(raw_corpus.keys())
N_DOCS      = len(ALL_DOC_IDS)
print(f'[LOAD] {N_DOCS} documents loaded from {BOOKS_DIR}')

# =============================================================================
#  SECTION 2 — PREPROCESSING
# =============================================================================

lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text: str) -> list:
    """Case-fold -> tokenise -> filter -> lemmatise -> stem."""
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


processed_corpus = {}
global_freq      = Counter()

for did in ALL_DOC_IDS:
    toks = preprocess_text(raw_corpus[did])
    processed_corpus[did] = toks
    global_freq.update(toks)

VOCABULARY = sorted(set(global_freq.keys()))
VOCAB_SET  = set(VOCABULARY)
print(f'[PREP] Vocabulary size: {len(VOCABULARY):,}')

# =============================================================================
#  SECTION 3 — INDEXING
# =============================================================================

# ── 3.1  Term-Document Incidence Matrix ──────────────────────────────────────
def build_tdim(processed_corpus, all_doc_ids, vocabulary):
    doc_token_sets = {did: set(processed_corpus[did]) for did in all_doc_ids}
    return {term: [1 if term in doc_token_sets[did] else 0 for did in all_doc_ids]
            for term in vocabulary}

t0 = time.time()
TDIM            = build_tdim(processed_corpus, ALL_DOC_IDS, VOCABULARY)
TDIM_BUILD_TIME = time.time() - t0
print(f'[IDX]  TDIM built in {TDIM_BUILD_TIME:.4f}s')

# ── 3.2  Inverted + Direct Index ─────────────────────────────────────────────
def build_inverted_and_direct_index(processed_corpus, all_doc_ids):
    inverted = defaultdict(list)
    direct   = defaultdict(list)
    for did in all_doc_ids:
        freq = Counter(processed_corpus[did])
        for term, tf in freq.items():
            inverted[term].append((did, tf))
            direct[did].append((term, tf))
    for term in inverted:
        inverted[term].sort(key=lambda x: x[0])
    return dict(inverted), dict(direct)

t0 = time.time()
INVERTED_INDEX, DIRECT_INDEX = build_inverted_and_direct_index(
    processed_corpus, ALL_DOC_IDS)
INV_BUILD_TIME = time.time() - t0
print(f'[IDX]  Inverted+Direct built in {INV_BUILD_TIME:.4f}s')

# ── 3.3  Binary Search Tree ───────────────────────────────────────────────────
class Node:
    def __init__(self, label, docs=None):
        self.label = label
        self.docs  = list(set(docs or []))
        self.left  = self.right = None

    def merge(self, node):
        self.docs = sorted(set(self.docs + node.docs))
        return self


class Tree:
    def __init__(self):
        self.root = None

    def insert(self, label, docs):
        new_node = Node(label, docs)
        if self.root is None:
            self.root = new_node
            return
        cur = self.root
        while True:
            if label == cur.label:
                cur.merge(new_node)
                return
            elif label < cur.label:
                if cur.left is None:
                    cur.left = new_node; return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = new_node; return
                cur = cur.right

    def search(self, label):
        cur = self.root
        while cur:
            if label == cur.label: return cur
            cur = cur.left if label < cur.label else cur.right
        return None


def build_bst_from_corpus(processed_corpus, all_doc_ids):
    bst = Tree()
    for did in all_doc_ids:
        for token in set(processed_corpus[did]):
            bst.insert(token, [did])
    return bst

t0 = time.time()
BST_INDEX      = build_bst_from_corpus(processed_corpus, ALL_DOC_IDS)
BST_BUILD_TIME = time.time() - t0
print(f'[IDX]  BST built in {BST_BUILD_TIME:.4f}s')

# =============================================================================
#  SECTION 4 — POSTING LIST MERGE HELPERS
# =============================================================================

def merge(p1, p2):
    answer, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:   answer.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:  i += 1
        else:                 j += 1
    return answer, 0


def union_postings(p1, p2):
    result, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:   result.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:  result.append(p1[i]); i += 1
        else:                 result.append(p2[j]); j += 1
    result.extend(p1[i:]); result.extend(p2[j:])
    return result


def not_postings(p, all_doc_ids):
    p_set = set(p)
    return [d for d in all_doc_ids if d not in p_set]

# =============================================================================
#  SECTION 5 — SPELL CORRECTION
# =============================================================================

def edit_distance(s1, s2):
    s1, s2 = str(s1), str(s2)
    l1, l2 = len(s1), len(s2)
    m = [[0]*(l2+1) for _ in range(l1+1)]
    for i in range(1, l1+1): m[i][0] = i
    for j in range(1, l2+1): m[0][j] = j
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            cost    = 0 if s1[i-1] == s2[j-1] else 1
            m[i][j] = min(m[i-1][j]+1, m[i][j-1]+1, m[i-1][j-1]+cost)
    return m[l1][l2]


def correct_spelling(term, vocab_set, vocabulary):
    if term in vocab_set:
        return term, False, 0
    best_term, best_dist = term, float('inf')
    for v in vocabulary:
        d = edit_distance(term, v)
        if d < best_dist:
            best_dist, best_term = d, v
    return best_term, True, best_dist

# =============================================================================
#  SECTION 6 — LOOKUP HELPERS
# =============================================================================

def lookup_tdim(term):
    vec = TDIM.get(term)
    return [] if vec is None else [ALL_DOC_IDS[i] for i, v in enumerate(vec) if v == 1]

def lookup_inverted(term):
    return [did for did, _ in INVERTED_INDEX.get(term, [])]

def lookup_bst(term):
    node = BST_INDEX.search(term)
    return node.docs if node else []

INDEX_LOOKUP = {'TDIM': lookup_tdim, 'INVERTED': lookup_inverted, 'BST': lookup_bst}

# =============================================================================
#  SECTION 7 — QUERY PARSER
# =============================================================================

PRECEDENCE = {'NOT': 3, 'AND': 2, 'OR': 1}
OPERATORS  = {'AND', 'OR', 'NOT'}


class QueryParser:
    def __init__(self, index_name='INVERTED'):
        assert index_name in INDEX_LOOKUP
        self.index_name  = index_name
        self.lookup      = INDEX_LOOKUP[index_name]
        self.corrections = []

    def preprocess_term(self, term):
        tokens = preprocess_text(term)
        return tokens[0] if tokens else term.lower()

    def get_postings(self, raw_term):
        processed = self.preprocess_term(raw_term)
        corrected, was_corrected, dist = correct_spelling(processed, VOCAB_SET, VOCABULARY)
        if was_corrected:
            self.corrections.append({'original': raw_term,
                                     'corrected': corrected,
                                     'distance': dist})
        return sorted(self.lookup(corrected))

    def tokenize_query(self, query):
        return re.sub(r'([()])', r' \1 ', query).split()

    def apply_operator(self, op, operands):
        if op == 'NOT':
            operands.append(not_postings(operands.pop(), ALL_DOC_IDS))
        else:
            right = operands.pop(); left = operands.pop()
            result, _ = merge(left, right) if op == 'AND' else (union_postings(left, right), 0)
            operands.append(sorted(result))

    def parse_tokens(self, tokens):
        operators, operands = [], []
        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    self.apply_operator(operators.pop(), operands)
                if operators: operators.pop()
                if operators and operators[-1] == 'NOT':
                    self.apply_operator(operators.pop(), operands)
            elif token in OPERATORS:
                while (operators and operators[-1] != '('
                       and operators[-1] in OPERATORS
                       and PRECEDENCE.get(operators[-1], 0) >= PRECEDENCE.get(token, 0)):
                    self.apply_operator(operators.pop(), operands)
                operators.append(token)
            else:
                operands.append(self.get_postings(token))
        while operators:
            self.apply_operator(operators.pop(), operands)
        return operands[0] if operands else []

    def parse_query(self, query, top_k=None):
        self.corrections = []
        t0      = time.time()
        tokens  = self.tokenize_query(query)
        result  = sorted(self.parse_tokens(tokens))
        elapsed = time.time() - t0
        display = result[:top_k] if (top_k and top_k > 0) else result
        return {
            'query'      : query,
            'index'      : self.index_name,
            'tokens'     : tokens,
            'corrections': self.corrections,
            'total'      : len(result),
            'showing'    : len(display),
            'results'    : [{'id': did, 'title': id_to_title[did]} for did in display],
            'elapsed_ms' : round(elapsed * 1000, 3),
        }

# =============================================================================
#  SECTION 8 — FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Shakespeare Search Engine</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0f0f14;
    --surface:  #1a1a24;
    --surface2: #22222f;
    --border:   #2e2e42;
    --accent:   #7c6af7;
    --accent2:  #a78bfa;
    --gold:     #f0b429;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --radius:   12px;
  }
  html { font-size: 15px; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 2rem 1rem 4rem;
  }
  header { text-align: center; margin-bottom: 2.5rem; }
  header h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent2), var(--gold));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  header p { color: var(--muted); margin-top: .4rem; font-size: .95rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.6rem 2rem;
    width: 100%; max-width: 860px; margin-bottom: 1.4rem;
  }
  .card h2 {
    font-size: .85rem; font-weight: 700; color: var(--accent2);
    margin-bottom: 1rem; text-transform: uppercase; letter-spacing: .08em;
  }
  .search-row { display: flex; gap: .7rem; flex-wrap: wrap; }
  input[type=text] {
    flex: 1 1 260px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: .65rem 1rem; font-size: 1rem;
    outline: none; transition: border-color .2s;
  }
  input[type=text]:focus { border-color: var(--accent); }
  select {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); padding: .65rem .8rem;
    font-size: .92rem; outline: none; cursor: pointer;
  }
  .btn {
    border: none; border-radius: 8px; cursor: pointer;
    font-weight: 700; font-size: .95rem; padding: .65rem 1.5rem;
    transition: opacity .15s;
  }
  .btn:hover { opacity: .85; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-ghost { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
  .controls { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; align-items: center; }
  .controls label { color: var(--muted); font-size: .88rem; }
  .controls select { font-size: .88rem; padding: .45rem .7rem; }
  .chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .9rem; }
  .chip {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: .25rem .75rem;
    font-size: .8rem; color: var(--muted); cursor: pointer;
    transition: background .15s, color .15s;
  }
  .chip:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
  .stats { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1.2rem; }
  .stat-pill {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: .3rem .85rem; font-size: .82rem;
  }
  .stat-pill span { color: var(--accent2); font-weight: 700; }
  .correction-box {
    background: #2a1f00; border: 1px solid var(--gold); border-radius: 8px;
    padding: .7rem 1rem; margin-top: 1rem; font-size: .88rem; color: var(--gold);
  }
  .correction-box strong { color: #ffe08a; }
  .result-list { list-style: none; margin-top: 1rem; display: flex; flex-direction: column; gap: .5rem; }
  .result-item {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: .65rem 1rem;
    display: flex; align-items: center; gap: .8rem; transition: border-color .2s;
  }
  .result-item:hover { border-color: var(--accent); }
  .result-rank { color: var(--muted); font-size: .82rem; font-weight: 700; min-width: 28px; text-align: right; }
  .result-title { font-weight: 600; flex: 1; }
  .result-id {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: .15rem .5rem; font-size: .78rem; color: var(--muted);
  }
  .info-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .8rem; }
  .info-box {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem; text-align: center;
  }
  .info-box .val { font-size: 1.4rem; font-weight: 800; color: var(--accent2); }
  .info-box .lbl { font-size: .8rem; color: var(--muted); margin-top: .2rem; }
  .empty { text-align: center; color: var(--muted); padding: 2rem 0; font-size: .95rem; }
  .error-msg { color: var(--red); font-size: .88rem; margin-top: .6rem; }
  .loader { display: none; }
  .loader.active { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin {
    width: 18px; height: 18px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite; display: inline-block; vertical-align: middle;
  }
  @media (max-width:600px) {
    header h1 { font-size: 1.5rem; }
    .info-grid { grid-template-columns: 1fr 1fr; }
    .card { padding: 1.2rem; }
  }
</style>
</head>
<body>

<header>
  <h1>Shakespeare Search Engine</h1>
  <p>Boolean retrieval over 44 Shakespeare works &nbsp;&mdash;&nbsp; TDIM &bull; Inverted Index &bull; BST</p>
</header>

<!-- Index stats card -->
<div class="card">
  <h2>Index Overview</h2>
  <div class="info-grid">
    <div class="info-box">
      <div class="val" id="stat-docs">—</div>
      <div class="lbl">Documents</div>
    </div>
    <div class="info-box">
      <div class="val" id="stat-vocab">—</div>
      <div class="lbl">Vocabulary Terms</div>
    </div>
    <div class="info-box">
      <div class="val" id="stat-time">—</div>
      <div class="lbl">Total Index Build Time (s)</div>
    </div>
  </div>
</div>

<!-- Search card -->
<div class="card">
  <h2>Boolean Query</h2>
  <div class="search-row">
    <input type="text" id="query-input"
           placeholder='e.g.  love AND death   |   king OR queen   |   NOT war'/>
    <button class="btn btn-primary" onclick="runSearch()">Search</button>
    <button class="btn btn-ghost"   onclick="clearAll()">Clear</button>
  </div>

  <div class="chips">
    <span class="chip" onclick="insertOp('AND')">AND</span>
    <span class="chip" onclick="insertOp('OR')">OR</span>
    <span class="chip" onclick="insertOp('NOT ')">NOT</span>
    <span class="chip" onclick="insertOp('(')">( </span>
    <span class="chip" onclick="insertOp(')')"> )</span>
    <span class="chip" onclick="setQuery('love AND death')">love AND death</span>
    <span class="chip" onclick="setQuery('king OR queen')">king OR queen</span>
    <span class="chip" onclick="setQuery('tragedy AND (death OR murder)')">tragedy AND (death OR murder)</span>
    <span class="chip" onclick="setQuery('NOT war')">NOT war</span>
    <span class="chip" onclick="setQuery('rome OR caesar AND NOT comedy')">rome OR caesar AND NOT comedy</span>
    <span class="chip" onclick="setQuery('hamlt AND denmarc')">hamlt AND denmarc ✱</span>
  </div>

  <div class="controls">
    <div>
      <label>Index &nbsp;</label>
      <select id="index-sel">
        <option value="INVERTED" selected>Inverted Index</option>
        <option value="TDIM">Term-Document Matrix</option>
        <option value="BST">Binary Search Tree</option>
      </select>
    </div>
    <div>
      <label>Show &nbsp;</label>
      <select id="topk-sel">
        <option value="5">Top 5</option>
        <option value="10" selected>Top 10</option>
        <option value="20">Top 20</option>
        <option value="30">Top 30</option>
        <option value="0">All</option>
      </select>
    </div>
    <span class="loader" id="loader"><span class="spin"></span>&nbsp;Searching…</span>
  </div>
  <div class="error-msg" id="error-msg"></div>
</div>

<!-- Results card -->
<div class="card" id="results-card" style="display:none">
  <h2>Results</h2>
  <div class="stats" id="stats-row"></div>
  <div id="correction-area"></div>
  <ul class="result-list" id="result-list"></ul>
</div>

<script>
const qInput  = document.getElementById('query-input');
const idxSel  = document.getElementById('index-sel');
const topkSel = document.getElementById('topk-sel');
const loader  = document.getElementById('loader');
const errMsg  = document.getElementById('error-msg');
const resCard = document.getElementById('results-card');

qInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

function insertOp(op) {
  qInput.focus();
  qInput.value = (qInput.value ? qInput.value.trimEnd() + ' ' : '') + op;
}
function setQuery(q) { qInput.value = q; runSearch(); }
function clearAll() { qInput.value = ''; resCard.style.display = 'none'; errMsg.textContent = ''; }

async function runSearch() {
  const query = qInput.value.trim();
  if (!query) return;
  errMsg.textContent = '';
  loader.classList.add('active');
  try {
    const resp = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, index: idxSel.value, top_k: parseInt(topkSel.value) })
    });
    const data = await resp.json();
    if (!resp.ok) { errMsg.textContent = data.error || 'Unknown error'; return; }
    renderResults(data);
  } catch(e) {
    errMsg.textContent = 'Network error: ' + e.message;
  } finally {
    loader.classList.remove('active');
  }
}

function renderResults(d) {
  resCard.style.display = 'block';
  document.getElementById('stats-row').innerHTML = `
    <div class="stat-pill">Found <span>${d.total}</span> doc(s)</div>
    <div class="stat-pill">Showing <span>${d.showing}</span></div>
    <div class="stat-pill">Index <span>${d.index}</span></div>
    <div class="stat-pill">Time <span>${d.elapsed_ms} ms</span></div>`;

  const corrArea = document.getElementById('correction-area');
  if (d.corrections && d.corrections.length) {
    const items = d.corrections.map(c =>
      `<strong>"${c.original}"</strong> &rarr; <strong>"${c.corrected}"</strong> (edit distance: ${c.distance})`
    ).join(' &nbsp;|&nbsp; ');
    corrArea.innerHTML = `<div class="correction-box">Spell correction applied: ${items}</div>`;
  } else {
    corrArea.innerHTML = '';
  }

  const ul = document.getElementById('result-list');
  if (!d.results.length) {
    ul.innerHTML = '<div class="empty">No documents matched this query.</div>';
    return;
  }
  ul.innerHTML = d.results.map((r, i) => `
    <li class="result-item">
      <span class="result-rank">#${i+1}</span>
      <span class="result-title">${esc(r.title)}</span>
      <span class="result-id">Doc ${r.id}</span>
    </li>`).join('');
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function loadStats() {
  try {
    const d = await (await fetch('/stats')).json();
    document.getElementById('stat-docs').textContent  = d.docs;
    document.getElementById('stat-vocab').textContent = d.vocab.toLocaleString();
    document.getElementById('stat-time').textContent  = d.total_index_time.toFixed(4);
  } catch(e) { console.warn('Stats failed', e); }
}
loadStats();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')


@app.route('/stats')
def stats():
    return jsonify({
        'docs'            : N_DOCS,
        'vocab'           : len(VOCABULARY),
        'tdim_time'       : round(TDIM_BUILD_TIME, 4),
        'inv_time'        : round(INV_BUILD_TIME, 4),
        'bst_time'        : round(BST_BUILD_TIME, 4),
        'total_index_time': round(TDIM_BUILD_TIME + INV_BUILD_TIME + BST_BUILD_TIME, 4),
    })


@app.route('/search', methods=['POST'])
def search():
    body       = request.get_json(force=True)
    query      = body.get('query', '').strip()
    index_name = body.get('index', 'INVERTED').upper()
    top_k      = int(body.get('top_k', 10))

    if not query:
        return jsonify({'error': 'Empty query'}), 400
    if index_name not in INDEX_LOOKUP:
        return jsonify({'error': f'Unknown index: {index_name}'}), 400

    try:
        parser = QueryParser(index_name=index_name)
        result = parser.parse_query(query, top_k=top_k if top_k > 0 else None)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('\n[APP]  Starting Flask server at http://localhost:5000')
    print('[APP]  Press Ctrl+C to quit.\n')
    app.run(debug=False, port=5000)
