import sys
import numpy as np
from scipy.linalg import block_diag

def parser():
  lines = [line.strip() for line in sys.stdin]
  def done():
    return len(lines) == 0
  def peek():
    return lines[0]
  def eat():
    nonlocal lines
    l = lines[0]
    lines = lines[1:]
    return l

  return peek, eat, done

peek, eat, done = parser()

while peek().startswith('*') or peek().startswith('"'):
  eat()

M = int(eat())
NBlocks = int(eat())
BlockStruct = [int(x) for x in eat().split()]
C = [int(x) for x in eat().split()]

fblks = []
for _ in range(M+1):
  blcks = []
  for s in BlockStruct:
    blcks.append(np.zeros((abs(s), abs(s))))
  fblks.append(blcks)

while not done():
  ln = eat()
  if ln == "":
    continue
  k, b, i, j, v = [int(x) for x in ln.split()]
  if BlockStruct[b - 1] < 0:
    assert(i == j)
  fblks[k][b - 1][i - 1, j - 1] = v
  fblks[k][b - 1][j - 1, i - 1] = v

Fs = [block_diag(*bs) for bs in fblks]

for bid in range(NBlocks):
  print(f'Constraint {bid}:')
  for i in range(1, M+1):
    print(f'F_{i}')
    print(fblks[i][bid])
  print(f'F_0')
  print(fblks[0][bid])
