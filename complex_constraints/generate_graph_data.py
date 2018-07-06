import sys
from argparse import ArgumentParser
from sets import Set
from collections import deque
from numpy.random import choice

class Node():
   def __init__(self, ind):
      self.neighbs = Set()
      self.ind = ind


class Grid():
   def __init__(self, n, nodes):
      # List of lists of nodes
      self.nodes = nodes

      # Make sure we're getting the right number of nodes for a grid
      assert len(nodes) == n*n

      # TODO: Adding bidirectional map from node pairs <-> edge numbers
      edgemap = {}
      for e in xrange(n*(n-1)*2):
         if e < n * (n-1):
            edgemap[(self.nodes[n*(e/(n-1)) + e%(n-1)],self.nodes[n*(e/(n-1)) + e%(n-1)+1])] = e
            edgemap[(self.nodes[n*(e/(n-1)) + e%(n-1)+1]),self.nodes[n*(e/(n-1)) + e%(n-1)]] = e
            edgemap[e] = (n*(e/(n-1)) + e%(n-1),n*(e/(n-1)) + e%(n-1)+1)
         else:
            e -= n*(n-1)
            edgemap[(self.nodes[n*(e%(n-1)) + e/(n-1)],self.nodes[n*(e%(n-1)+1) + e/(n-1)])] = e + n*(n-1)
            edgemap[(self.nodes[n*(e%(n-1)+1) + e/(n-1)],self.nodes[n*(e%(n-1)) + e/(n-1)])] = e + n*(n-1)
            edgemap[e + n*(n-1)] = (n*(e%(n-1)) + e/(n-1), n*(e%(n-1)+1) + e/(n-1))

      self.edgemap = edgemap

   def connected_components(self):
      """ Get a list of the connected components that form this graph, as a list of sets of nodes"""
      # List of sets of components
      comps = []
      # We're going to floodfill, need to keep track of what's visited
      visited = Set()
      # Floodfill, attempting to start from each element
      for node in self.nodes:
         if not node in visited:
            # If it's not visited, create a new component
            comp = Set()
            # Fill
            stack = [node]
            visited.add(node)
            while stack:
               curr = stack.pop()
               comp.add(curr)
               for neighb in curr.neighbs:
                  if not neighb in visited:
                     stack.append(neighb)
                     # Want to mark it here so same node doesn't get added twice
                     visited.add(neighb)

            comps.append(comp)

      return comps

   def bfs(self, start, end):
      """ Finds all shortest paths from start to end, returns this as a list of lists"""
      queue = deque()
      queue.append((start,))
      sols = []
      while queue:
         curr = queue.popleft()
         # If this is the destination, we should add it to list of solutions and stop with this paths
         if curr[-1] == end:
            sols.append(curr)
            continue
         # If we've found a solution already and we've passed the length of that solution, time to convert to edges and quit
         if sols and len(curr) > len(sols[0]): return self.toEdges(sols)
         # Otherwise, add all neighbours as per usual
         for node in curr[-1].neighbs:
            queue.append(curr + (node,))

      # Don't actually want a list of nodes, want a list of edge numbers
      return self.toEdges(sols)

   def toEdges(self, sols):
      """ Convert a list of lists of nodes representing a path in a list of lists of edge numbers """
      return [[self.edgemap[(l[i],l[i+1])] for i in xrange(len(l)-1)] for l in sols]

def script(out, n):
    #grid = gen_grid_removed(3, [])
    #sols = grid.bfs(grid.nodes[0], grid.nodes[-1])
    #print sols
    #print gen_grid_removed(3, [0, 1, 2, 4, 5]).connected_components()
    with open(out, 'w') as file:
      for q in xrange(n):
         # Randomly remove 1/3 of edges
         to_remove = choice(24, 0, replace=False)
         graph_write = '-'.join([str(x) for x in to_remove])
         print to_remove
         # Generate grid and connected components
         grid = gen_grid_removed(4, to_remove)
         comps = grid.connected_components()
         data = []
         for comp in comps:
            if len(comp) < 5: continue
            allpairs = pairs(list(comp))
            chosen = choice(len(allpairs), len(allpairs)/3, replace=False)
            chosen = [allpairs[c] for c in chosen]
            for n1, n2 in chosen:
               nodes_write = '-'.join([str(n1.ind), str(n2.ind)])
               sols_write = ['-'.join([str(x) for x in p]) for p in grid.bfs(n1, n2)]
               file.write(','.join([graph_write] + [nodes_write] + sols_write) + '\n')
            data += [(p, grid.bfs(p[0], p[1])) for p in chosen]



def gen_grid_removed(n, removed):
   """
   Generate a nxn grid, with the listed edge indices removed. The edges are ordered from left to right, then top to bottom, then the same thing sideways.
   """
   grid = gen_grid(n)

   for e in removed:
      # First half are intra-row
      n1, n2 = grid.edgemap[e]
      unlink(grid.nodes[n1], grid.nodes[n2])

   return grid

def gen_grid(n):
   """ Generate nxn grid """
   rows = [[Node(j + n*i) for j in xrange(n)] for i in xrange(n)]

   for i in xrange(n):
      for j in xrange(n-1):
         # Connections within rows
         link(rows[i][j], rows[i][j+1])
         # Connections between rows
         link(rows[j][i], rows[j+1][i])

   # Flatten nodes and create grid
   nodes = [x for l in rows for x in l]
   return Grid(n, nodes)

def link(n1, n2):
   """ Connect 2 nodes n1 and n2 by adding them to each other's adjacency lists"""
   n1.neighbs.add(n2)
   n2.neighbs.add(n1)

def unlink(n1, n2):
   """ Remove 2 assumed neighbours by removing the from each other's adjacency lists"""
   n1.neighbs.remove(n2)
   n2.neighbs.remove(n1)

def pairs(l):
   """ Return a list of tuples representing all possible pairs of elements in l """
   ret = []
   for i in xrange(len(l)):
      for j in xrange(i+1, len(l)):
         ret.append((l[i], l[j]))
   return ret

def parse_args(args):
   parser = ArgumentParser()

   # Arguments go here
   parser.add_argument('out', help='Where to output data')
   parser.add_argument('n', type=int, help='Number of random graph configurations to use')

   return parser.parse_args(args)


def main(args=sys.argv[1:]):
   args = parse_args(args)
   script(**vars(args))

if __name__ == '__main__':
   sys.exit(main())

