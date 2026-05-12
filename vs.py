import numpy as np

def calcuate_interaction_m(u: int, v: int, lmbda: float, z: float):
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
        L_e.append(calcuate_interaction_m(u, v, f, z[i]))

    return np.vstack(L_e)

# find derivative and proportional for PD controller
# use first order approx for kd
def calculate_pd(e_history, dt):
    p = e_history[-1]
    if len(e_history) >= 2:
        d = (e_history[-1] - e_history[-2]) / dt
    else:
        d = np.zeros_like(p)
    return p, d
    
# eye in hand control
# outputs joint velocities (6 x 1) given L_e (interaction matrix, 2k x 6), e_history (vector of errors in each feature, 2k x n)
# k (list of PD proportionality constants), and v (list of past velocities)
def eih_find_velocity(L_e, params: list, e_history: list):
    # np pinv function automatically calculates psuedo inverse
    psuedoinv = np.linalg.pinv(L_e)
    p, d = calculate_pd(e_history, params[1])
    return psuedoinv @ (params[0][0]*p + params[0][1]*d)

# control loop with simulation placeholders
# control_params includes max iterations (0), focal length of camera, (1), kp/kd constants (3), and delta t (4)
def ibvs_control(get_current_pts, get_depth, desired_pt, control_params):
    e_history = []
    for i in range(control_params[0]): 
        cur = get_current_pts()
        z = get_depth()
        e_history.append((cur - desired_pt).flatten())
        if(np.linalg.norm(e_history[-1]) <= control_params[3]):
            break
        L_e = find_interaction(cur, control_params[1], z) 
        eih_find_velocity(L_e, control_params[2:3], e_history) 


