import sys
import os
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag
from itertools import product
from math import sqrt

EPS = 1e-6

Parser = namedtuple('Parser', ['peek', 'eat', 'done'])
S = '₀₁₂₃₄₅₆₇₈₉'

def ss(n):
  s = [S[ord(c)-ord('0')] for c in str(n)]
  return ''.join(s)

def build_parser(fname):
  with open(fname) as file:
    lines = [line.strip() for line in file.read().split('\n')]

  def done():
    return len(lines) == 0
  def peek():
    return lines[0]
  def eat():
    nonlocal lines
    l = lines[0]
    lines = lines[1:]
    return l

  return Parser(peek, eat, done)

if len(sys.argv) <= 1:
  print('Missing args', file=sys.stderr)
  sys.exit(1)

Meta = namedtuple('Meta', ['m', 'n_blocks', 'block_struct'])

def read_meta(parser):
  peek, eat, done = parser

  while peek().startswith('*') or peek().startswith('"'):
    eat()

  m = int(eat())
  n_blocks = int(eat())
  block_struct = [int(x) for x in eat().split()]

  return Meta(m, n_blocks, block_struct)

def read_body(parser, meta, expect_ints=False):

  c = [float(x) for x in parser.eat().split()]

  fblks = []
  for _ in range(meta.m + 1):
    blcks = []
    for s in meta.block_struct:
      blcks.append(np.zeros((abs(s), abs(s)), dtype=np.int32 if expect_ints else np.float64))
    fblks.append(blcks)

  while not parser.done():
    ln = parser.eat()
    if ln == "":
      continue
    k, b, i, j, v = ln.split()
    k, b, i, j = [int(x) for x in (k, b, i, j)]
    v = int(v) if expect_ints else float(v)
    if meta.block_struct[b - 1] < 0:
      assert(i == j)
    fblks[k][b - 1][i - 1, j - 1] = v
    fblks[k][b - 1][j - 1, i - 1] = v

  return c, fblks

def proc_body(meta, c, blcks):
  linear_ineqs = []
  cs_ineqs = []
  for bid in range(meta.n_blocks): # Constraint bid
    linear = meta.block_struct[bid] < 0
    if linear:
      for s in range(abs(meta.block_struct[bid])):
        coef_pairs = []
        for i in range(1, meta.m+1):
          coef = blcks[i][bid][s, s]
          if abs(coef) < EPS:
            continue
          coef_pairs.append((blcks[i][bid][s, s], i))

        bound = blcks[0][bid][s, s]
        linear_ineqs.append((coef_pairs, bound))
      continue

    coef_pairs = []
    for i in range(1, meta.m+1):
      if not np.any(blcks[i][bid]):
        continue
      coef_pairs.append((blcks[i][bid], i))
    bound = blcks[0][bid]
    assert(not np.any(bound))
    cs_ineqs.append(coef_pairs)
  return linear_ineqs, cs_ineqs

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

def print_body(meta, c, linear_ineqs, cs_ineqs, cert_coefs):
  print('Objective: Minimise ', end='')
  s = [f'{v:.6f}F{ss(i+1)}' for i, v in enumerate(c) if abs(v) > EPS]
  print(' + '.join(s))

  for cid, (coef_pairs, bound) in enumerate(linear_ineqs):
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
        print('λ', coef)
    shape = coef_pairs[0][0].shape
    assert(shape[0] == shape[1])
    for rw in range(shape[0]):
      for col in range(shape[0]):
        pairs = [(mat[rw, col], i) for (mat, i) in coef_pairs if abs(mat[rw, col])> EPS]
        print(f'{format_coefs(pairs):^15}',end='')
      print()

def sumup(meta, obj, linear_ineqs, cs_ineqs, linear_coefs, cs_coefs):
  sm = np.zeros(meta.m)
  λ = 0
  for idx, (linear_ineq, bound) in enumerate(linear_ineqs):
    vec = np.zeros(meta.m)
    for (coef, i) in linear_ineq:
      vec[i - 1] = coef
    sm += linear_coefs[idx] * vec
    λ += linear_coefs[idx] * bound

  for idx, cs_ineq in enumerate(cs_ineqs):
    for (mat, i) in cs_ineq:
      sm[i - 1] += (cs_coefs[idx] * mat).sum()

  print(abs(sm -np.array(obj)).sum() < EPS)
  print(abs(sm -np.array(obj)).sum())
  assert(abs(λ + 120/4)< EPS)

def proc_problem():
  parser = build_parser(sys.argv[1])

  meta = read_meta(parser)
  c, blcks = read_body(parser, meta, expect_ints=True)
  linear_ineqs, cs_ineqs = proc_body(meta, c, blcks)

  #print_body(meta, c, linear_ineqs, cs_ineqs, None)

  assert(len(sys.argv) >= 3)
  cert_parser = build_parser(sys.argv[2])
  _, cert_blcks = read_body(cert_parser, Meta(2, meta.n_blocks, meta.block_struct))
  linear_coefs, cs_coefs = proc_cert(meta, cert_blcks)

  linear_ineqs = [
    linear_ineqs[59],
    linear_ineqs[60],
    linear_ineqs[61],
    linear_ineqs[62],
    linear_ineqs[63],
    linear_ineqs[70]]

  # collapsed middle eqns
  eqn8 = ([(1, 37), (2, 55), (-2, 16), (-1, 17)], 0)
  eqn12 = ([(-2, 40), (-2, 55), (2, 18)], 0)

  linear_ineqs.append(eqn8)
  linear_ineqs.append(eqn12)

  # Clean and scale objective function
  c = [0]*meta.m
  c[8] = -2
  c[36] = -1
  c[54] = -2

  # Clear coefs
  linear_coefs = [0]*len(linear_ineqs)
  cs_coefs = [0]*2

  # Adjust equality
  linear_coefs[0] = 6
  linear_coefs[1] = 1
  linear_coefs[2] = 0.25
  linear_coefs[3] = 2
  linear_coefs[4] = 1
  linear_coefs[5] = 1
  linear_coefs[6] = 2
  linear_coefs[7] = 2

  va1 = np.array([0, 0, -2, 1/2, 1/2, 1])
  va1 /= sqrt(11/2)
  la1 = 11/8
  va2 = np.array([0, 0, 0, -1.0, 1, 0])
  va2 /= sqrt(2)
  la2 = 1/8
  cs_coefs[0] = la1 * np.outer(va1, va1) + la2 * np.outer(va2, va2)

  vb1 = np.array([0, 0,  1/2 * (-11 + sqrt(89)), 1, 1/2 * (7-sqrt(89)), 0, 1])
  vb1 /= sqrt(89 - 9 * sqrt(89))
  lb1 = 1/4 * (11 + sqrt(89))
  vb2 = np.array([0, 0, 1/2 * (-11 - sqrt(89)), 1, 1/2 * (7 + sqrt(89)), 0, 1])
  vb2 /= sqrt(89 + 9 * sqrt(89))
  lb2 = 1/4 * (11 - sqrt(89))
  cs_coefs[1] = lb1 * np.outer(vb1, vb1) + lb2 * np.outer(vb2, vb2)

  sumup(meta, c, linear_ineqs, cs_ineqs, linear_coefs, cs_coefs)
  print_body(meta, c, linear_ineqs, cs_ineqs, (linear_coefs, cs_coefs))

  with np.printoptions(precision=3, suppress=True):
    print(lb1 * np.outer(vb1, vb1))
    print(lb2 * np.outer(vb2, vb2))
    print(lb1 * np.outer(vb1, vb1) + lb2 * np.outer(vb2, vb2))

  # sm = np.zeros(meta.m)
  # for idx in range(8):
  #   (linear_ineq, bound) = linear_ineqs[idx]
  #   vec = np.zeros(meta.m)
  #   for (coef, i) in linear_ineq:
  #     vec[i - 1] = coef
  #   sm += linear_coefs[idx] * vec
  # prs = []
  # for i in range(meta.m):
  #   if abs(sm[i]) > EPS:
  #     prs.append((sm[i], i+1))
  # print(format_coefs(prs))


proc_problem()
