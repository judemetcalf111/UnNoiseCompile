import numpy as np
from functools import reduce


def create_kron_matrix(n, a0_vals, a1_vals):
    """
    Constructs the transition matrix using Kronecker products.
    """
    # 1. Build the list of 2x2 matrices for each bit position
    matrices = []
    
    for k in range(n):
        # Column 0: Source is 0. Rows: [Stay 0, Flip to 1]
        col0 = [1 - a0_vals[k], a0_vals[k]]
        
        # Column 1: Source is 1. Rows: [Flip to 0, Stay 1]
        col1 = [a1_vals[k],     1 - a1_vals[k]]
        
        # Stack into a 2x2 matrix
        # [[1-a0,  a1],
        #  [ a0, 1-a1]]
        m_k = np.array([col0, col1]).T 
        matrices.append(m_k)

    # 2. Compute the Kronecker product of all matrices in sequence
    # M0 (x) M1 (x) M2 ...
    full_matrix = reduce(np.kron, matrices)
    
    return full_matrix

# --- Example Usage ---

n = 3
a0 = np.array([0.1, 0.2, 0.3])
a1 = np.array([0.5, 0.6, 0.7])

result = create_kron_matrix(n, a0, a1)

print(f"Shape: {result.shape}")
print("Top-left 4x4 block:")
print(np.round(result[:4, :4], 3))

# Verification of M[1,0] (from 000 to 001)
# 0->0 (1-0.1), 0->0 (1-0.2), 0->1 (0.3)
# 0.9 * 0.8 * 0.3 = 0.216
print(f"\nValue at [1,0]: {result[1,0]:.4f}")

def fast_interaction_multiply(vector, n, a0_vals, a1_vals):
	"""
	Computes (M . vector) without creating the matrix M.
	Complexity: O(n * 2^n)
	"""
	# Working copy of the vector (must be float for probability calcs)
	v = np.asarray(vector, dtype = np.float64)
	# v = vector.astype(np.float64)
	
	# Iterate through each bit position (from MSB to LSB)
	# Corresponding to the Kronecker sequence M_0 (x) M_1 ...
	for k in range(n):
		# 1. Construct the small 2x2 transition matrix for bit k
		# Cols: [Source 0, Source 1]
		col0 = [1 - a0_vals[k], a0_vals[k]]
		col1 = [a1_vals[k],     1 - a1_vals[k]]
		m_k = np.array([col0, col1]).T  # Shape (2, 2)
		
		# 2. Reshape vector to isolate bit k
		# The vector of length 2^n is reshaped into (Prefix, Bit_k, Suffix)
		# Prefix count: 2^k (Bits processed so far)
		# Bit_k count:  2   (The dimension we are transforming)
		# Suffix count: 2^(n-1-k) (Bits remaining)
		dim_prefix = 2**k
		dim_suffix = 2**(n - 1 - k)
		
		v_reshaped = v.reshape((dim_prefix, 2, dim_suffix))
		
		# 3. Apply the 2x2 matrix to the middle dimension (axis 1)
		# We use Einstein Summation for clarity and speed.
		# 'ijk' is input tensor (prefix, bit, suffix)
		# 'lb' is the small matrix (row l, col b)
		# 'ilk' is output tensor (prefix, new_bit, suffix)
		# Essentially: dot product along the 'bit' axis
		v_transformed = np.einsum('lb, ibk -> ilk', m_k, v_reshaped)
		
		# 4. Flatten back to 1D for the next iteration
		v = v_transformed.flatten()
	
	return v

# --- Verification ---

# Setup parameters (Same as previous example)
n = 10  # Let's try a larger n where full matrix is 1024x1024
counts = np.random.randint(1, 100, 2**n) # Random initial vector

a0 = np.random.rand(n)
a1 = np.random.rand(n)

# 1. Fast Method
result_fast = fast_interaction_multiply(counts, n, a0, a1)

# 2. Slow Method (Verification using Kronecker)
# Only feasible for small n, doing n=10 is okay (1M entries)
# but don't try this for n=20!
from functools import reduce
matrices = []
for k in range(n):
    c0 = [1 - a0[k], a0[k]]
    c1 = [a1[k],     1 - a1[k]]
    matrices.append(np.array([c0, c1]).T)
full_matrix = reduce(np.kron, matrices)

result_slow = full_matrix @ counts

print(f"Vector Size: {2**n}")
print(f"Fast Result (First 5): {np.round(result_fast[:5], 2)}")
print(f"Slow Result (First 5): {np.round(result_slow[:5], 2)}")

# Check if they are identical
is_close = np.allclose(result_fast, result_slow)
print(f"\nDo results match? {is_close}")
