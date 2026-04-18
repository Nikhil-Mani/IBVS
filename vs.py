import numpy as np

def calculate_matrix(u: int, v: int, lmbda: float, z: float):
    # find the interaction matrix for one feature
    return np.array([lmbda/z, 0, -u/z, -(u*v/z), (lmbda**2 + u**2)/lmbda, -v], 
                    [0, lmbda/z, -v/z, -(lmbda**2 - v**2)/lmbda, (u*v)/lmbda, u])

def find_interaction(points: list, f, z):
    # points is list of k (u,v) pairs
    # f is focal len
    # z is list of k points
    L_e = []
    for i, (u, v) in enumerate(points):
        L_e.append(calculate_matrix(u, v, f, z[i]))

    return np.vstack([L_e])
# this function will solve for dr given du, dv, and an interaction matrix
def find_velocity(df: list, J, m):
    # check the dimensionality of k and m
    # k = rows / 2
    k = J.shape[0] / 2
    Jinv = np.linalg.inv(J)
    # J inverse exists!
    if(k==m):
        r = Jinv @ df
    if(k > m):
        # left psuedo inverse
        psuedoinv = np.linalg.inv(J.T @ J) @ J.T
        r = psuedoinv @ df
    if(k < m):
        # right psuedo inverse
        psuedoinv = J.T @ np.linalg.inv(J.T @ J) 
        r = psuedoinv @ df
    return r


