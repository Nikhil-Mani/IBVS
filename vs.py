import numpy as np

def calcuate_interaction_m(u: int, v: int, lmbda: float, z: float):
    # find the interaction matrix for one feature
    return np.array([
        [lmbda/z, 0, -u/z, -(u*v/lmbda), (lmbda**2 + u**2)/lmbda, -v], 
        [0, lmbda/z, -v/z, -(lmbda**2 + v**2)/lmbda, (u*v)/lmbda, u]
                    ])

# returns L_e (interaction matrix, 2k x 6)
def find_interaction(points: list, f, z, cx, cy):
    # points is list of k (u,v) pairs
    # f is focal len
    # z is list of k points
    L_e = []
    for i, (u_raw, v_raw) in enumerate(points):
        u = u_raw - cx
        v = v_raw - cy
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
# outputs joint velocities (6 x 1) given L_e (interaction matrix, 2k x 6), 
# e_history (vector of errors in each feature, 2k x n)
# k (list of PD proportionality constants), and dt (delta time)
def eih_find_velocity(L_e, k, dt, e_history: list):
    # np pinv function automatically calculates psuedo inverse
    psuedoinv = np.linalg.pinv(L_e)
    p, d = calculate_pd(e_history, dt)
    return psuedoinv @ (k[0]*p + k[1]*d)
GLOBAL_E_HISTORY = []
# control loop with simulation placeholders
# control_params includes max iterations (0), focal length of camera, (1), 
# kp/kd constants (2), delta t (3), and threshold (4)
def ibvs_control(current_pts, depth_map, desired_pt, control_params):
    global GLOBAL_E_HISTORY
    cur = current_pts
    #get depth for feature pts
    z_depths = []
    for (u, v) in cur:
        z_depths.append(depth_map[int(u), int(v)])
    error = cur - desired_pt.flatten()
    GLOBAL_E_HISTORY.append((cur - desired_pt).flatten())
    if(len(GLOBAL_E_HISTORY) > 10):
        GLOBAL_E_HISTORY.pop(0)
    if(np.linalg.norm(error) <= control_params[3]):
        return np.zeros(6)
    cx, cy = 320.0, 240.0
    L_e = find_interaction(cur, control_params[0], z_depths, cx, cy) 
    vel = eih_find_velocity(L_e, control_params[1], control_params[2], GLOBAL_E_HISTORY)

    return vel


