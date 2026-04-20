import numpy as np

def calculate_matrix(u: int, v: int, lmbda: float, z: float):
    # find the interaction matrix for one feature
    return np.array([
        [lmbda/z, 0, -u/z, -(u*v/lmbda), (lmbda**2 + u**2)/lmbda, -v], 
        [0, lmbda/z, -v/z, -(lmbda**2 + v**2)/lmbda, (u*v)/lmbda, u]
                    ])

# returns L_e (interaction matrix, 2k x 6)
def find_interaction(points: list, f, z):
    # points is list of k (u,v) pairs
    # f is focal len
    # z is list of k points
    L_e = []
    for i, (u, v) in enumerate(points):
        L_e.append(calculate_matrix(u, v, f, z[i]))

    return np.vstack(L_e)

# eye in hand control
# outputs joint velocities (6 x 1) given L_e (interaction matrix, 2k x 6), e (vector of errors in each feature, 2k x 1)
# k (list of PID proportionality constants), and v (list of past velocities)
# TODO: finish PID control
def eih_find_velocity(e: list, L_e, k: list, v: list):
    # np pinv function automatically calculates psuedo inverse
    psuedoinv = np.linalg.pinv(L_e)
    return k[0] * psuedoinv @ e

# eye to hand control
# same as above, but also requires robot jacobian and V, translation matrix
# assume at tool ce
def eth_find_velocity(df: list, L_e, J, k: list, e, V):
    psuedoinv = np.linalg.pinv(L_e @ V @ J)
    return k[0] * psuedoinv @ e



