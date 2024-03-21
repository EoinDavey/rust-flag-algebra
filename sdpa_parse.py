import sys
import os
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag

Parser = namedtuple('Parser', ['peek', 'eat', 'done'])

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

def read_meta(parser):
  peek, eat, done = parser

  while peek().startswith('*') or peek().startswith('"'):
    eat()

  m = int(eat())
  n_blocks = int(eat())
  block_struct = [int(x) for x in eat().split()]

  return m, n_blocks, block_struct

def read_body(parser, m, n_blocks, block_struct):
  peek, eat, done = parser

  c = [float(x) for x in eat().split()]

  fblks = []
  for _ in range(m+1):
    blcks = []
    for s in block_struct:
      blcks.append(np.zeros((abs(s), abs(s))))
    fblks.append(blcks)

  while not done():
    ln = eat()
    if ln == "":
      continue
    k, b, i, j = [int(x) for x in ln.split()[:-1]]
    v = float(ln.split()[-1])
    if block_struct[b - 1] < 0:
      assert(i == j)
    fblks[k][b - 1][i - 1, j - 1] = v
    fblks[k][b - 1][j - 1, i - 1] = v

  return c, fblks

def print_body(c, blcks, m, n_blocks):
  print(f'C: {c}')

  fs = [block_diag(*bs) for bs in blcks]

  for i in range(1, len(fs)):
    print(f'F_{i}:\n{fs[i]}')
  print(f'F_{0}:\n{fs[0]}')

  #for bid in range(n_blocks):
  #  print(f'Constraint {bid}:')
  #  for i in range(1, m+1):
  #    print(f'F_{i}')
  #    print(blcks[i][bid])
  #  print(f'F_0')
  #  print(blcks[0][bid])

def print_soln(c, blcks):
  print(f'coefs:', [f'{x:.2f}' for x in c])

  fs = [block_diag(*bs) for bs in blcks]
  with np.printoptions(precision=3, suppress=True):
    print(fs[2])

def proc_problem():
  parser = build_parser(sys.argv[1])

  m, n_blocks, block_struct = read_meta(parser)
  c, blcks = read_body(parser, m, n_blocks, block_struct)
  print_body(c, blcks, m, n_blocks)

  if len(sys.argv) < 3:
    return

  cert_parser = build_parser(sys.argv[2])
  cert_c, cert_blcks = read_body(cert_parser, 2, n_blocks, block_struct)
  print_soln(cert_c, cert_blcks)

proc_problem()
