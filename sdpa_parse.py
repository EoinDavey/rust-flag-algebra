import sys
import numpy as np
from collections import namedtuple, deque
from scipy import sparse

if len(sys.argv) <= 1:
  print('Usage: sdpa_parse <sdpa_file> [certificate_file]', file=sys.stderr)
  sys.exit(1)

# Small epsilon value, under which we consider a number to be zero.
EPS = 1e-6

# Small function to convert integers to subscript string.
def ss(n):
  S = '₀₁₂₃₄₅₆₇₈₉'
  return ''.join([S[ord(c)-ord('0')] for c in str(n)])

# Simple parser class to read files
class Parser:
  def __init__(self, fname):
    with open(fname) as file:
      lines = [line.strip() for line in file.read().split('\n')]
    self.lines = deque(lines)

  def done(self):
    return len(self.lines) == 0
  def peek(self):
    return self.lines[0]
  def eat(self):
    return self.lines.popleft()

# Object to store metadata
Meta = namedtuple('Meta', ['m', 'n_blocks', 'block_struct'])

# Read metadata from SDPA-Sparse file
def read_meta(parser):
  while parser.peek().startswith('*') or parser.peek().startswith('"'):
    parser.eat()

  m = int(parser.eat())
  n_blocks = int(parser.eat())
  block_struct = [int(x) for x in parser.eat().split()]

  return Meta(m, n_blocks, block_struct)

# Read main body of an SDPA-Sparse file
def read_body(parser, meta, expect_ints=False):
  c = [float(x) for x in parser.eat().split()]

  fblks = [[None] * meta.n_blocks for _ in range(meta.m+1)]

  while not parser.done():
    ln = parser.eat()
    if ln == "":
      continue
    k, b, i, j, v = ln.split()
    k, b, i, j = [int(x) for x in (k, b, i, j)]
    v = int(v) if expect_ints else float(v)
    if meta.block_struct[b - 1] < 0:
      assert(i == j)
    if fblks[k][b - 1] == None:
      fblks[k][b - 1] = sparse.dok_array((abs(meta.block_struct[b-1]), abs(meta.block_struct[b-1])), dtype=np.int32 if expect_ints else np.float64)
    fblks[k][b - 1][i - 1, j - 1] = v
    fblks[k][b - 1][j - 1, i - 1] = v

  for k in range(meta.m + 1):
    for b in range(meta.n_blocks):
      if fblks[k][b] != None:
        fblks[k][b] = fblks[k][b].tocsr()

  return c, fblks

# proc_body takes the raw blocks from an SDPA-Sparse file and converts
# them to a format which represents the underlying inequalities.
def proc_body(meta, c, blcks):
  linear_ineqs = []
  for bid in range(meta.n_blocks): # Constraint bid
    linear = meta.block_struct[bid] < 0 # SDPA-Parse format trickery, negative if diagonal.
    if not linear:
      continue
    coef_pairs = [[] for _ in range(abs(meta.block_struct[bid]))]
    for i in range(1, meta.m+1):
      if blcks[i][bid] == None:
        continue
      rs, _ = blcks[i][bid].nonzero()
      for r in rs:
        v = blcks[i][bid][r, r]
        coef_pairs[r].append((v, i))
    bounds = np.zeros(abs(meta.block_struct[bid])) if blcks[0][bid] == None else blcks[0][bid].diagonal()
    linear_ineqs += list(zip(coef_pairs, bounds))

  cs_ineqs = []
  for bid in range(meta.n_blocks): # Constraint bid
    linear = meta.block_struct[bid] < 0 # SDPA-Parse format trickery, negative if diagonal.
    if linear:
      continue
    coef_pairs = []
    for i in range(1, meta.m+1):
      if blcks[i][bid] == None or blcks[i][bid].count_nonzero() == 0:
        continue
      coef_pairs.append((blcks[i][bid], i))
    bound = blcks[0][bid]
    assert(bound == None or bound.count_nonzero() == 0) # We assume CS ineqs are all >=0 ineqs
    cs_ineqs.append(coef_pairs)

  return linear_ineqs, cs_ineqs

# proc_cert converts the raw blocks of a certificate into coefficients for
# our inequalities. We only need the dual solution so most of the certificate is
# ignored.
def proc_cert(meta, cert_blcks):
  linear_coefs = []
  cs_coefs = []
  for bid in range(meta.n_blocks): # Constraint bid
    linear = meta.block_struct[bid] < 0
    if linear:
      for s in range(abs(meta.block_struct[bid])):
        linear_coefs.append(cert_blcks[2][bid][s, s])
      continue

    cs_coefs.append(cert_blcks[2][bid])
  return linear_coefs, cs_coefs

# Print a vector nicely
def format_coefs(coef_pairs):
  o = ""
  for i, (c, idx) in enumerate(coef_pairs):
    if i != 0:
      o += ' + ' if c > 0 else ' - '
    if i == 0 and c < 0:
      o += '-'
    c = abs(c)
    o += f'F{ss(idx)}' if abs(c-1) < EPS else f'{c}F{ss(idx)}'
  return o

# Take a processed SDPA-Parse file, and optionally a certificate, and print
# nicely.
def print_body(meta, c, linear_ineqs, cs_ineqs, cert_coefs=None):
  print('Objective: Minimise ', end='')
  s = [f'{v:.6f}F{ss(i+1)}' for i, v in enumerate(c) if abs(v) > EPS]
  print(' + '.join(s))

  for cid, (coef_pairs, bound) in enumerate(linear_ineqs):
    # If we have the corresponding coefficient, print it and skip if its zero.
    if cert_coefs != None:
      coef = cert_coefs[0][cid]
      if abs(coef) < EPS:
        continue
      print(f'Linear Constraint {cid}. λ = {coef}')
    else:
      print(f'Linear Constraint {cid}:')
    print(format_coefs(coef_pairs), end='')
    print(f' >= {bound}')

  for idx, coef_pairs in enumerate(cs_ineqs):
    print(f'CS constraint {idx}')
    if cert_coefs != None:
      coef = cert_coefs[1][idx]
      with np.printoptions(precision=3, suppress=True):
        print('λ', coef.todense())
    shape = coef_pairs[0][0].shape
    assert(shape[0] == shape[1])
    # Collect coefficient matrices and print in a table.
    for rw in range(shape[0]):
      for col in range(shape[0]):
        pairs = [(mat[rw, col], i) for (mat, i) in coef_pairs if abs(mat[rw, col])> EPS]
        print(f'{format_coefs(pairs):^15}',end='')
      print()

# Take a set of inequalities and their coefficients and compute
# the corresponding bound for the objective function.
def sumup(meta, obj, linear_ineqs, cs_ineqs, cert_coefs):
  linear_coefs, cs_coefs = cert_coefs
  sm = np.zeros(meta.m)
  total_coef = 0
  for idx, (linear_ineq, bound) in enumerate(linear_ineqs):
    vec = np.zeros(meta.m)
    for (coef, i) in linear_ineq:
      vec[i - 1] = coef
    sm += linear_coefs[idx] * vec
    total_coef += linear_coefs[idx] * bound

  for idx, cs_ineq in enumerate(cs_ineqs):
    for (mat, i) in cs_ineq:
      sm[i - 1] += (cs_coefs[idx] * mat).sum()

  assert(abs(sm -np.array(obj)).sum() < EPS)
  return total_coef

def proc_problem():
  parser = Parser(sys.argv[1])

  meta = read_meta(parser)
  c, blcks = read_body(parser, meta, expect_ints=False)
  print('Read Body')
  linear_ineqs, cs_ineqs = proc_body(meta, c, blcks)
  print('Processed Body')

  if len(sys.argv) >= 3:
    cert_parser = Parser(sys.argv[2])
    _, cert_blcks = read_body(cert_parser, Meta(2, meta.n_blocks, meta.block_struct))
    cert_coefs = proc_cert(meta, cert_blcks)

    print_body(meta, c, linear_ineqs, cs_ineqs, cert_coefs)
    print('Value:', sumup(meta, c, linear_ineqs, cs_ineqs, cert_coefs))
  else:
    print_body(meta, c, linear_ineqs, cs_ineqs, None)

proc_problem()
