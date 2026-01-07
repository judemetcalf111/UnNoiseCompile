import sys
import collections
import random

# # Data Style
# graph_data = {
#     '2': ['6'],
#     '6': ['2', '5', '7', '12'],
#     '3': ['4', '9'],
#     #...
# }

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
            
            # Warn if degree > max_sets (Need more than 4 cicuits to cover)
            if len(neighbors) > max_sets:
                print(f"WARNING: Qubit {u} has {len(neighbors)} connections. It cannot be fully covered by {max_sets} Circuits.")

            for v in neighbors:
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                
                # Build undirected edge list
                if (p1, p2) not in seen:
                    self.all_edges.append((str(p1), str(p2)))
                    seen.add((p1, p2))
                    # Initialize for colouring
                    self.adj[str(p1)][str(p2)] = None
                    self.adj[str(p2)][str(p1)] = None
                
                # Build fast adjacency for density checks
                self.raw_adj[str(u)].add(str(v))
                self.raw_adj[str(v)].add(str(u))

    # KEMPE CHAIN colouring
    def get_free_colours(self, u):
        used = {self.adj[u][v] for v in self.adj[u] if self.adj[u][v] is not None}
        return set(range(1, self.max_sets + 1)) - used

    def get_kempe_chain(self, start_node, c1, c2):
        chain = []
        curr = start_node
        seeking = c1 
        while True:
            found = False
            for neighbor, colour in self.adj[curr].items():
                if colour == seeking:
                    chain.append((curr, neighbor, colour))
                    curr = neighbor
                    seeking = c2 if seeking == c1 else c1
                    found = True
                    break
            if not found: break
        return chain

    def swap_chain(self, chain, c1, c2):
        for u, v, colour in chain:
            new_colour = c2 if colour == c1 else c1
            self.adj[u][v] = new_colour
            self.adj[v][u] = new_colour

    def colour_initial_edges(self):
        random.shuffle(self.all_edges)  # Randomize input order
        
        untested = set()                # Edges that couldn't be coloured initially. 
                                        # The graph-theory problem is actually NP-Complete, but here we add another circuit making the problem O(E) = O(N).

        for u, v in self.all_edges:
            free_u = self.get_free_colours(u)
            free_v = self.get_free_colours(v)
            common = free_u.intersection(free_v)
            
            if common:
                c = list(common)[0]
                self.adj[u][v] = c
                self.adj[v][u] = c
            else:
                # Kempe Swap needed
                if not free_u or not free_v: 
                    # Adding the edge to untested set
                    untested.add((u, v))
                    continue

                c_u = list(free_u)[0]
                c_v = list(free_v)[0]
                chain = self.get_kempe_chain(v, c_u, c_v)
                
                if not chain or chain[-1][1] != u:
                    # Failed swap
                    untested.add((u, v))
                    continue
                else:
                    self.swap_chain(chain, c_u, c_v)
                    self.adj[u][v] = c_u
                    self.adj[v][u] = c_u

        # Extract initial sparse sets
        sets = [set() for _ in range(self.max_sets)]
        for u, neighbors in self.adj.items():
            for v, colour in neighbors.items():
                if int(u) < int(v) and colour is not None:
                    sets[colour-1].add((u, v))

        # Adding any 5th 'Untested' Circuit:
        if untested:
            print(f"INFO: {len(untested)} edges could not be assigned within {self.max_sets} circuits. Adding an extra circuit to cover them.")
            sets.append(untested)
        return sets

    # Density Maximisation to maximise the qubits tested on any given run
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
        for _, s in enumerate(valid_sets):
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

class MisraGriesSolver:
    def __init__(self, raw_data, max_sets=4):
        self.adj = collections.defaultdict(dict)  # self.adj[u][v] = colour
        self.edges_to_colour = []
        self.max_sets = max_sets
        
        # --- Pre-processing ---
        seen = set()
        for u, neighbors in raw_data.items():
            for v in neighbors:
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                
                # Store edge
                if (p1, p2) not in seen:
                    self.edges_to_colour.append((str(p1), str(p2)))
                    seen.add((p1, p2))
                
                # Initialize adj structure
                if str(v) not in self.adj[str(p1)]: self.adj[str(p1)][str(v)] = None
                if str(p1) not in self.adj[str(v)]: self.adj[str(v)][str(p1)] = None

    def get_free_colours(self, u, max_colours):
        """Returns a set of colours NOT used by node u."""
        used = {self.adj[u][n] for n in self.adj[u] if self.adj[u][n] is not None}
        return {c for c in range(1, max_colours + 1) if c not in used}

    def invert_path(self, u, v, c_u, c_v):
        """Inverts a 2-coloured path (Kempe Chain) starting at u."""
        path = []
        curr = u
        seeking = c_u
        other = c_v
        
        # Build path
        while True:
            found_next = False
            for neighbor, colour in self.adj[curr].items():
                if colour == seeking and neighbor != v: # Don't go back to v immediately
                    # prevent cycles or immediate backtracking logic if needed, 
                    # but simple traversal usually works for Kempe
                    if (curr, neighbor) not in path and (neighbor, curr) not in path:
                        path.append((curr, neighbor))
                        curr = neighbor
                        seeking, other = other, seeking
                        found_next = True
                        break
            if not found_next:
                break
        
        # Swap colours
        for n1, n2 in path:
            current_c = self.adj[n1][n2]
            new_c = c_v if current_c == c_u else c_u
            self.adj[n1][n2] = new_c
            self.adj[n2][n1] = new_c

    def solve(self):
        max_degree = 0
        for u in self.adj:
            max_degree = max(max_degree, len(self.adj[u]))

        if max_degree > self.max_sets:
            raise ValueError("Graph degree exceeds maximum allowed sets.")

        # Vizing's theorem guarantees solution with Delta + 1 colours
        max_colours = max_degree + 1
        print(f"Graph Degree: {max_degree}. Solving with max {max_colours} circuits.")

        for x, y in self.edges_to_colour:
            # 1. Find free colours for u (x) and v (y)
            free_x = list(self.get_free_colours(x, max_colours))
            free_y = list(self.get_free_colours(y, max_colours))
            
            # Case 1: Easy match
            common = set(free_x).intersection(free_y)
            if common:
                c = list(common)[0]
                self.adj[x][y] = c
                self.adj[y][x] = c
                continue

            # Case 2: The Fan Argument (Misra & Gries)
            # If no common colour, we must swap.
            # For simplicity in this implementation, we use the standard "Kempe Swap" 
            # which is the subroutine inside Vizing's proof.
            
            c_x = free_x[0]
            c_y = free_y[0]
            
            # Try to swap colours on the path starting from y using colours (c_x, c_y)
            # This frees up c_x at node y
            self.invert_path(y, x, c_x, c_y)
            
            # Now assign c_x to the edge (x, y)
            self.adj[x][y] = c_x
            self.adj[y][x] = c_x

        # Extract Sets
        sets = [set() for _ in range(max_colours)]
        for u, neighbors in self.adj.items():
            for v, colour in neighbors.items():
                if int(u) < int(v) and colour is not None:
                    sets[colour-1].add((u, v))
        
        # Filter out empty sets (if graph was actually Class 1)
        return [s for s in sets if s]
    

# Increase recursion depth just in case, though small circuits won't hit this.
sys.setrecursionlimit(5000)

class ExactGraphSolver:
    def __init__(self, raw_data, max_sets=4):
        self.max_sets = max_sets
        self.adj = {} # Adjacency: self.adj[u][v] = colour
        self.all_edges = []
        
        # --- PRE-PROCESSING ---
        seen = set()
        for u, neighbors in raw_data.items():
            if u not in self.adj: self.adj[u] = {}
            
            for v in neighbors:
                if v not in self.adj: self.adj[v] = {}
                
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                u_str, v_str = str(p1), str(p2)
                
                if (u_str, v_str) not in seen:
                    self.all_edges.append((u_str, v_str))
                    seen.add((p1, p2))
                    # Initialize as None (uncoloured)
                    self.adj[u_str][v_str] = None
                    self.adj[v_str][u_str] = None

        # Heuristic: Sort edges by "Degree Sum". 
        # Hardest edges (connecting two busy hubs) are processed first to fail fast.
        self.all_edges.sort(key=lambda pair: len(self.adj[pair[0]]) + len(self.adj[pair[1]]), reverse=True)

    def is_valid(self, u, v, colour):
        """Checks if assigning 'colour' to (u,v) causes a conflict."""
        # Check u's existing edges
        for neighbor, c in self.adj[u].items():
            if c == colour: return False
        # Check v's existing edges
        for neighbor, c in self.adj[v].items():
            if c == colour: return False
        return True

    def backtrack(self, edge_index):
        # Base Case: All edges successfully coloured
        if edge_index == len(self.all_edges):
            return True

        u, v = self.all_edges[edge_index]

        # Try colours 1 through 4
        for colour in range(1, self.max_sets + 1):
            if self.is_valid(u, v, colour):
                # Apply colour
                self.adj[u][v] = colour
                self.adj[v][u] = colour
                
                # Recurse to next edge
                if self.backtrack(edge_index + 1):
                    return True
                
                # Backtrack (Undo)
                self.adj[u][v] = None
                self.adj[v][u] = None
        
        # If no colour works for this edge given previous choices, return False
        return False

    def solve(self):
        print(f"Attempting to solve exact matching for {len(self.all_edges)} edges...")
        success = self.backtrack(0)
        
        if not success:
            raise ValueError("Exact solution impossible with 4 sets! (Is the graph truly Class 1?)")
            
        # Format output
        sets = [set() for _ in range(self.max_sets)]
        for u, neighbors in self.adj.items():
            for v, colour in neighbors.items():
                if int(u) < int(v):
                    sets[colour-1].add((u, v))
        return sets
