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

def verify_coverage(raw_data, final_sets):
    """
    Verifies that every connection in 'raw_data' appears at least once 
    in 'final_sets'.
    
    Returns:
        missing_edges (list): A list of edges that were NEVER tested.
        coverage_stats (dict): Stats on how well the graph is covered.
    """
    # 1. Reconstruct the Truth (All expected unique edges)
    expected_edges = set()
    for u, neighbors in raw_data.items():
        for v in neighbors:
            # Normalize edge to (min, max) to handle undirected nature
            u_int, v_int = int(u), int(v)
            p1, p2 = min(u_int, v_int), max(u_int, v_int)
            expected_edges.add((str(p1), str(p2)))
            
    # 2. Flatten the generated circuits into a single set of tested edges
    tested_edges = set()
    total_tests = 0
    for s in final_sets:
        for u, v in s:
            u_int, v_int = int(u), int(v)
            p1, p2 = min(u_int, v_int), max(u_int, v_int)
            tested_edges.add((str(p1), str(p2)))
            total_tests += 1

    # 3. Find Diff
    missing_edges = list(expected_edges - tested_edges)
    
    # 4. Reporting
    circuit_count = len(final_sets)
    total_expected = len(expected_edges)
    unique_tested = len(tested_edges)
    coverage_percent = (unique_tested / total_expected) * 100 if total_expected > 0 else 0.0
    
    print("-" * 40)
    print(f"VERIFICATION REPORT")
    print("-" * 40)
    print(f"Total Number of Circuits:    {circuit_count}")
    print(f"Total Unique Edges in Graph: {total_expected}")
    print(f"Unique Edges Tested:         {unique_tested}")
    print(f"Total Tests Performed:       {total_tests} (Avg {total_tests/total_expected:.2f}x redundancy)" if total_expected else "N/A")
    print(f"Coverage:                    {coverage_percent:.1f}%")
    
    if missing_edges:
        print(f"\n[FAIL] {len(missing_edges)} edges are MISSING:")
        for edge in missing_edges:
            print(f"  - Edge {edge}")
    else:
        print("\n[PASS] All connections tested at least once.")
    print("-" * 40)

    return missing_edges

def compact_sets(sets):
    """
    Post-processing to maximize density in earlier sets.
    Tries to move edges from Set 5 -> Set 1, Set 4 -> Set 2, etc.
    """
    # Flatten sets into a list of mutable sets
    layers = [set(s) for s in sets]
    num_layers = len(layers)

    changed = True
    while changed:
        changed = False
        # Iterate backwards: Try to empty the last set into the first set
        for i in range(num_layers - 1, 0, -1):      # Source Layer (e.g., 4)
            for j in range(i):                       # Target Layer (e.g., 0, 1, 2, 3)
                
                # Identify candidates to move from Layer i to Layer j
                # An edge (u, v) can move to Layer j if neither u nor v are busy in Layer j
                
                # Build 'busy' map for target layer j for fast lookup
                busy_nodes = {n for pair in layers[j] for n in pair}
                
                # Find moveable edges
                movable = []
                for u, v in layers[i]:
                    if u not in busy_nodes and v not in busy_nodes:
                        movable.append((u, v))
                        # Update busy_nodes temporarily so we don't move conflicting edges at once
                        busy_nodes.add(u)
                        busy_nodes.add(v)
                
                # Perform moves
                if movable:
                    for edge in movable:
                        layers[i].remove(edge)
                        layers[j].add(edge)
                    changed = True

    # Remove empty sets from the end (if any were fully emptied)
    return [s for s in layers if s]

def fill_with_redundancy(all_edges, current_sets):
        """
        Fills empty slots in existing sets with redundant edges.
        PRIORITY: Edges that have been tested the least so far.
        """
        # 1. Global Counter: How many times has each edge been tested?
        edge_counts = collections.defaultdict(int)
        for s in current_sets:
            for edge in s:
                edge_counts[edge] += 1
        
        # Ensure all edges are in the counter (even if count is 0, though it shouldn't be)
        for edge in all_edges:
            if edge not in edge_counts:
                edge_counts[edge] = 0

        # 2. Iterate through each set to fill 'holes'
        for s in current_sets:
            # Identify which qubits are currently busy in this circuit
            busy_nodes = {n for u, v in s for n in (u, v)}
            
            # Find all valid candidates for this specific circuit
            candidates = []
            for u, v in all_edges:
                # Rule 1: Edge must not already be in this specific circuit
                if (u, v) in s:
                    continue
                
                # Rule 2: Both Qubits must be free in this circuit
                if u not in busy_nodes and v not in busy_nodes:
                    candidates.append((u, v))
            
            # 3. Smart Sort: "Least Tested First"
            # We shuffle first to break ties randomly (ensuring uniform distribution)
            random.shuffle(candidates)
            candidates.sort(key=lambda e: edge_counts[e])
            
            # 4. Greedy Fill
            for u, v in candidates:
                # Re-check availability (since a previous candidate might have taken the spot)
                if u not in busy_nodes and v not in busy_nodes:
                    s.add((u, v))
                    
                    # Mark nodes as busy for this circuit
                    busy_nodes.add(u)
                    busy_nodes.add(v)
                    
                    # Increment global count so next circuits favor other edges
                    edge_counts[(u, v)] += 1
                    
        return current_sets



class AdvancedGraphSolver:
    def __init__(self, raw_data, max_sets=5):
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
    def __init__(self, raw_data, max_sets=5):
        self.adj = collections.defaultdict(dict)  # self.adj[u][v] = colour
        self.all_edges = []
        self.max_sets = max_sets
        
        # --- Pre-processing ---
        seen = set()
        for u, neighbors in raw_data.items():
            for v in neighbors:
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                
                # Store edge
                if (p1, p2) not in seen:
                    self.all_edges.append((str(p1), str(p2)))
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
            max_degree = max(max_degree, len(self.adj[u]) - 1)  # Degree is number of edges

        if max_degree > self.max_sets:
            raise ValueError("Graph degree exceeds maximum allowed sets.")

        # Vizing's theorem guarantees solution with Delta + 1 colours
        max_colours = max_degree + 1
        print(f"Graph Degree: {max_degree}. Solving with max {max_colours} circuits.")

        for x, y in self.all_edges:
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
    

# Increase recursion depth
# Only to be used on small QPUs due to performance.
sys.setrecursionlimit(500000)

class MinConflictsSolver:
    def __init__(self, raw_data, max_sets=4, max_steps=100000):
        self.max_sets = max_sets
        self.max_steps = max_steps
        self.adj = collections.defaultdict(dict)
        self.all_edges = []
        
        # --- Pre-processing ---
        seen = set()
        for u, neighbors in raw_data.items():
            # Initialize nodes in adjacency dict to ensure safe lookups later
            if str(u) not in self.adj: self.adj[str(u)] = {}
            
            for v in neighbors:
                u_int, v_int = int(u), int(v)
                p1, p2 = min(u_int, v_int), max(u_int, v_int)
                u_str, v_str = str(p1), str(p2)
                
                # Ensure neighbor structure exists
                if v_str not in self.adj: self.adj[v_str] = {}

                # Store edge in standardized format
                if (u_str, v_str) not in seen:
                    self.all_edges.append((u_str, v_str))
                    seen.add((p1, p2))
                    
                    # Update adjacency structure
                    # We store None initially; the coloring logic uses a separate dict
                    self.adj[u_str][v_str] = None
                    self.adj[v_str][u_str] = None

    def count_conflicts(self, u, v, color, current_coloring):
        """Counts how many neighbors would conflict if we assign 'color' to (u,v)."""
        conflicts = 0
        
        # Check u's neighbors (excluding v)
        for neighbor in self.adj[u]:
            if neighbor == v: continue
            # Construct the key for the neighbor edge
            edge_key = tuple(sorted((u, neighbor)))
            if current_coloring.get(edge_key) == color:
                conflicts += 1
        
        # Check v's neighbors (excluding u)
        for neighbor in self.adj[v]:
            if neighbor == u: continue
            # Construct the key for the neighbor edge
            edge_key = tuple(sorted((v, neighbor)))
            if current_coloring.get(edge_key) == color:
                conflicts += 1
                
        return conflicts

    def solve(self):
        print(f"Repairing solution for {len(self.all_edges)} edges using Min-Conflicts...")
        
        # 1. Initialization: Assign Random Colors to all edges
        # We store coloring in a simple dict for fast access: {(u, v): color}
        current_coloring = {}
        for u, v in self.all_edges:
            current_coloring[(u, v)] = random.randint(1, self.max_sets)

        # 2. Iterative Repair Loop
        for step in range(self.max_steps):
            
            # Find all currently conflicting edges
            conflicted_edges = []
            for u, v in self.all_edges:
                c = current_coloring[(u, v)]
                if self.count_conflicts(u, v, c, current_coloring) > 0:
                    conflicted_edges.append((u, v))
            
            # SUCCESS: No conflicts left!
            if not conflicted_edges:
                # print(f"Solved in {step} steps.") # Optional logging
                return self.format_output(current_coloring)
            
            # Pick a random conflicted edge to fix
            u, v = random.choice(conflicted_edges)
            
            # Find the color that minimizes conflicts for this edge
            min_conflicts = float('inf')
            candidates = []
            
            for color in range(1, self.max_sets + 1):
                c_score = self.count_conflicts(u, v, color, current_coloring)
                if c_score < min_conflicts:
                    min_conflicts = c_score
                    candidates = [color]
                elif c_score == min_conflicts:
                    candidates.append(color)
            
            # Assign best color (break ties randomly)
            new_color = random.choice(candidates)
            
            # RANDOM WALK: 
            # With small probability (5%), pick a completely random color to escape local loops
            if min_conflicts > 0 and random.random() < 0.05:
                 new_color = random.randint(1, self.max_sets)
            
            current_coloring[(u, v)] = new_color

        raise RuntimeError(f"MinConflictsSolver failed to find a solution in {self.max_steps} steps. \n"
                           f"Possible causes: Graph is not Class 1, or max_steps is too low for this graph size.")

    def format_output(self, coloring):
        # Convert the coloring dict back into the list-of-sets format expected by generate_circuits
        sets = [set() for _ in range(self.max_sets)]
        for (u, v), color in coloring.items():
            sets[color-1].add((u, v))
        return sets


def generate_circuits(solver_class, raw_data, max_sets=5, compaction=True, redundancy=True):
        """
        Parameters:
        - solver_class: A class reference to the solver to use (AdvancedGraphSolver, ExactGraphSolver, MisraGriesSolver).
        - raw_data: The raw graph data in adjacency list format.
        - max_sets: Maximum number of sets (circuits) to use.

        Generates an optimal schedule using the specified solver class.
        Steps:
        1. Solve: Use the provided solver to get initial sets.
        2. Compact: Rearrange sets to maximize early density.
        3. Redundancy: Fill in gaps with least-tested edges.
        4. Return final sets.

        Returns:
        final_sets: A list of sets representing the optimal schedule.
        """
        # Solver: 
        solver = solver_class(raw_data, max_sets)
        sets = solver.solve() 
        
        # Compaction: (Efficiency)
        if compaction:
            sets = compact_sets(sets)
        
        # Redundancy: Fill the circuit gaps with extra tests (Statistics)
        if redundancy:
            sets = fill_with_redundancy(solver.all_edges, sets)
        
        missing = verify_coverage(raw_data, sets)

        if missing:
            raise RuntimeError(f'Critical: The solver failed to cover the full graph!\n{missing} are not covered.')

        return sets


