import networkx as nx
from braket.aws import AwsDevice
from braket.devices import Devices
%matplotlib inline
import collections
import random
import time
import networkx as nx

# Data Style
graph_data = {
    '2': ['6'],
    '6': ['2', '5', '7', '12'],
    '3': ['4', '9'],
    #...
}

class AdvancedGraphSolver:
    def __init__(self, raw_data, max_sets=4):
        self.max_sets = max_sets
        self.adj = collections.defaultdict(dict) 
        self.all_edges = []
        self.raw_adj = collections.defaultdict(set) # Fast lookup for maximization
        
        seen = set()
        
        # --- PRE-PROCESSING ---
        for u, neighbors in raw_data.items():
            if not neighbors: continue # Ignore true islands
            
            # Warn if degree > max_sets (Pigeonhole principle violation)
            if len(neighbors) > max_sets:
                print(f"WARNING: Node {u} has {len(neighbors)} connections. It cannot be fully covered by {max_sets} sets.")

            for v in neighbors:
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                
                # Build undirected edge list
                if (p1, p2) not in seen:
                    self.all_edges.append((str(p1), str(p2)))
                    seen.add((p1, p2))
                    # Initialize for coloring
                    self.adj[str(p1)][str(p2)] = None
                    self.adj[str(p2)][str(p1)] = None
                
                # Build fast adjacency for density checks
                self.raw_adj[str(u)].add(str(v))
                self.raw_adj[str(v)].add(str(u))

    # --- PART 1: KEMPE CHAIN COLORING (Valid Partition) ---
    def get_free_colors(self, u):
        used = {self.adj[u][v] for v in self.adj[u] if self.adj[u][v] is not None}
        return set(range(1, self.max_sets + 1)) - used

    def get_kempe_chain(self, start_node, c1, c2):
        chain = []
        curr = start_node
        seeking = c1 
        while True:
            found = False
            for neighbor, color in self.adj[curr].items():
                if color == seeking:
                    chain.append((curr, neighbor, color))
                    curr = neighbor
                    seeking = c2 if seeking == c1 else c1
                    found = True
                    break
            if not found: break
        return chain

    def swap_chain(self, chain, c1, c2):
        for u, v, color in chain:
            new_color = c2 if color == c1 else c1
            self.adj[u][v] = new_color
            self.adj[v][u] = new_color

    def color_initial_edges(self):
        random.shuffle(self.all_edges) # Randomize input order
        
        for u, v in self.all_edges:
            free_u = self.get_free_colors(u)
            free_v = self.get_free_colors(v)
            common = free_u.intersection(free_v)
            
            if common:
                c = list(common)[0]
                self.adj[u][v] = c
                self.adj[v][u] = c
            else:
                # Kempe Swap needed
                if not free_u or not free_v: continue
                c_u = list(free_u)[0]
                c_v = list(free_v)[0]
                chain = self.get_kempe_chain(v, c_u, c_v)
                if not chain or chain[-1][1] != u:
                    self.swap_chain(chain, c_u, c_v)
                    self.adj[u][v] = c_u
                    self.adj[v][u] = c_u

        # Extract initial sparse sets
        sets = [set() for _ in range(self.max_sets)]
        for u, neighbors in self.adj.items():
            for v, color in neighbors.items():
                if int(u) < int(v) and color is not None:
                    sets[color-1].add((u, v))
        return sets

    # --- PART 2: DENSITY MAXIMIZATION (Augmenting Paths) ---
    def improve_matching(self, current_set_pairs):
        """Tries to rescue 'stranded' nodes by flipping edges (A-B, C-D) <-> (A-C, B-D)"""
        # Convert to dictionary for easy traversal
        partner = {}
        for u, v in current_set_pairs:
            partner[u] = v
            partner[v] = u
            
        nodes_in_graph = list(self.raw_adj.keys())
        random.shuffle(nodes_in_graph) # Randomize to spread improvements
        
        improvements = 0
        
        for u in nodes_in_graph:
            if u in partner: continue # Already matched
            
            # 'u' is free. Look for a path u -- v -- w -- z where 'z' is also free.
            neighbors = list(self.raw_adj[u])
            random.shuffle(neighbors)
            
            found_swap = False
            for v in neighbors:
                if v in partner:
                    w = partner[v]
                    # Check w's neighbors for a free node z
                    w_neighbors = list(self.raw_adj[w])
                    random.shuffle(w_neighbors)
                    
                    for z in w_neighbors:
                        if z != v and z not in partner:
                            # Swap found!
                            del partner[v]
                            del partner[w]
                            partner[u] = v
                            partner[v] = u
                            partner[w] = z
                            partner[z] = w
                            
                            improvements += 1
                            found_swap = True
                            break
                if found_swap: break
        
        # Reconstruct set
        new_set = set()
        seen = set()
        for u, v in partner.items():
            p1, p2 = min(int(u), int(v)), max(int(u), int(v))
            if (p1, p2) not in seen:
                new_set.add((str(p1), str(p2)))
                seen.add((p1, p2))
        return new_set, improvements

    def maximize_sets(self, valid_sets):
        final_sets = []
        for i, s in enumerate(valid_sets):
            current_set = set(s)
            
            # Loop a few times to iteratively improve
            for _ in range(3):
                # 1. Greedy Fill (Fastest)
                occupied = {n for pair in current_set for n in pair}
                random.shuffle(self.all_edges)
                for u, v in self.all_edges:
                    if (u,v) not in current_set and u not in occupied and v not in occupied:
                        current_set.add((u,v))
                        occupied.add(u)
                        occupied.add(v)
                
                # 2. Augmenting Path Swap (Smarter)
                current_set, improved = self.improve_matching(current_set)
                if improved == 0: break
            
            final_sets.append(current_set)
        return final_sets
