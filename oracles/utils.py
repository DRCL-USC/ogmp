import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scipy as sc
import enum
import matplotlib.pyplot as plt

class legs(enum.Enum):
    FL = 0
    FR = 1

# robot physical length paramters
L = 0.247  # length of body, measured from hip to shoulder
W = 0.194  # width between left and right feet 
L_thigh = 0.2  
L_calf = 0.2 

# position of corners of robot, in body frame (so it's a constant)
B_p_Bi = {}
B_p_Bi[legs.FL] = np.array([0, W / 2.0, -L / 2.0])
B_p_Bi[legs.FR] = np.array([0, -W / 2.0, -L / 2.0])

C_p_Bi = {}
C_p_Bi[legs.FL] = np.array([0, W / 2.0, L / 2.0])
C_p_Bi[legs.FR] = np.array([0, -W / 2.0, L / 2.0])


def square_wave_gen(n,N):
    seq = np.zeros(N)
    for m in range(1,int(n+1)):
        seq[int((m-1)*N/n):int(0.5*(2*m-1)*N/n)] = 1
    return seq

# function to calculate the inverse term frequenting in rde
def riccati_inverse(r,b,k): 
    return np.linalg.inv(r+np.matmul(b.T,np.matmul(k,b)))

# function to solve RDE
def riccati_solve(a,b,q,R):      
    N = 49
    k = q
    for i in range(N):
        k = np.matmul(a.T,np.matmul(k,a)) - np.matmul(a.T,np.matmul(k,b))@riccati_inverse(R,b,k)@np.matmul(b.T,np.matmul(k,a))+q
    
    return k

#returns 3x3 skew symmetric cross product matrix
def vec_2_skmat(vector):        
    v1 = vector[0]
    v2 = vector[1]
    v3 = vector[2]
    skmat = np.array([[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]])
    return skmat

# 2D rotation matrix
def rot_mat_2d_np(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


# given axis and angle, returns 3x3 rotation matrix
def rot_mat_np(s, th):
    # normalize s if isn't already normalized
    norm_s = np.linalg.norm(s)
    assert norm_s != 0.0
    s_normalized = s / norm_s

    # Rodrigues' rotation formula
    skew_s = vec_2_skmat(s_normalized)
    return np.eye(3) + np.sin(th) * skew_s + (1.0 - np.cos(th)) * skew_s @ skew_s

# converts euler angles to rotation matrix
def rot_from_rpy(roll,pitch,yaw):
    rot = Rot.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_matrix()

def rpy_from_flat_rot(R_flat):
    R = np.reshape(R_flat, (3, 3), order="F")
    r =  Rot.from_matrix(R)
    ro,pi,y =  r.as_euler("zyx")
    return ro,pi,y



# given position vector and rotation matrix, returns 4x4 homogeneous
# transformation matrix
def homog_np(p, R):
    return np.block([[R, p[:, np.newaxis]], [np.zeros((1, 3)), 1]])

# reverses the direction of the coordinate transformation defined by a 4x4
# homogeneous transformation matrix
def reverse_homog_np(T):
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog = np.zeros((4, 4))
    reverse_homog[:3, :3] = R.T
    reverse_homog[:3, 3] = -R.T @ p
    reverse_homog[3, 3] = 1.0
    return reverse_homog

# multiplication between a 4x4 homogenous transformation matrix and 3x1
# position vector, returns 3x1 position
def mult_homog_point_np(T, p):
    p_aug = np.concatenate((p, [1.0]))
    return (T @ p_aug)[:3]

# multiplication between a 4x4 homogenous transformation matrix and 3x1
# force vector, returns 3x1 force
def mult_homog_vec_np(T, f):
    f_aug = np.concatenate((f, [0.0]))
    return (T @ f_aug)[:3]


# generic planar 2 link inverse kinematics implementation
# returns the closest point within the workspace if the requested point is
# outside of it
def planar_IK_np(l1, l2, x, y, elbow_up):
    l = np.sqrt(x**2.0 + y**2.0)
    l = max(abs(l1 - l2), min(l, l1 + l2))

    alpha = np.arctan2(y, x)

    cos_beta = (l**2 + l1**2 - l2**2.0) / (2.0 * l * l1)
    cos_beta = max(-1.0, min(cos_beta, 1.0))
    beta = np.arccos(cos_beta)

    cos_th2_abs = (l**2 - l1**2.0 - l2**2.0) / (2.0 * l1 * l2)
    cos_th2_abs = max(-1.0, min(cos_th2_abs, 1.0))
    th2_abs = np.arccos(cos_th2_abs)

    if elbow_up:
        th1 = alpha - beta
        th2 = th2_abs
    else:
        th1 = alpha + beta
        th2 = -th2_abs

    return th1, th2


# generic planar 2 link jacobian inverse transpose calculation implementation
# end_effector_force = jacobian_inv_tranpose * joint_torque
def planar_jac_inv_transpose_np(l1, l2, th1, th2, tau1, tau2):
    J = np.array(
        [
            [-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
            [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)],
        ]
    )
    force = np.linalg.solve(J.T, np.array([tau1, tau2]))
    return force


# generic planar 2 link jacobian transpose calculation implementation
# joint_torque = jacobian_tranpose * end_effector_force
# end_effector_force is force that robot exerts on environment
def planar_jac_transpose_np(l1, l2, th1, th2, f1, f2):
    J = np.array(
        [
            [-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
            [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)],
        ]
    )
    tau = J.T @ np.array([f1, f2])
    return tau


# Solo specific functions below

# position of corners of robot, in body frame (so it's a constant)
B_T_Bi = {}
for leg in legs:
    B_T_Bi[leg] = homog_np(B_p_Bi[leg], np.eye(3))




def state_space(state_vector,control_vector,A,B):                  # forward prediction of new state
        
    return A@state_vector+ B@control_vector


    
def convert2draw_vec(state_vector,foot_pos,control):               
    p_i_flat = np.zeros(6); p_i_flat[:3] = foot_pos[0]; p_i_flat[3:6] = foot_pos[1] 
    X_k = state_vector[:12].flatten()
    U_k = np.hstack((p_i_flat, control[:].flatten()))
    return X_k, U_k

def convert2draw_LD(state_vector,foot_pos,control):
    
    p = state_vector[0:3].flatten()
    Ori= np.zeros(3)
    pdot = state_vector[3:6].flatten()
    omega = np.zeros(3)
    
    p_i_flat = np.zeros(6); p_i_flat[:3] = foot_pos[0]; p_i_flat[3:6] = foot_pos[1]
    f_i_flat = np.zeros(6); f_i_flat[:3] = control[:3].flatten(); f_i_flat[3:6] = control[3:].flatten()    
    X_k = np.hstack((Ori, p, omega, pdot))
    U_k = np.hstack((p_i_flat, f_i_flat))
    return X_k, U_k

def convert2draw_AD(state_vector,foot_pos,control):               
    p_i_flat = np.zeros(6); p_i_flat[:3] = foot_pos[0]; p_i_flat[3:6] = foot_pos[1] 
    X_k = state_vector[:12].flatten()
    U_k = np.hstack((p_i_flat, control[:].flatten()))
    return X_k, U_k


# given numpy trajectory matrix, extract state at timestep k
# note the order argument in reshape, which is necessary to make it consistent
# with casadi's reshape
def extract_state_np(X, U, k):
    p = X[3:6, k]
    # R_flat = X[3:12, k]
    # R = np.reshape(R_flat, (3, 3), order="F")
    Ori = X[:3,k]
    pdot = X[9:12, k]
    omega = X[6:9, k]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = U[3 * leg.value : leg.value * 3 + 3, k]
        f_i[leg] = U[6 + 3 * leg.value : 6 + leg.value * 3 + 3, k]
    return Ori,p,omega,pdot, p_i, f_i

def flatten_state_np(Ori, p, omega, pdot, p_i, f_i):
    #R_flat = np.reshape(R, 9, order="F")
    p_i_flat = np.zeros(12)
    f_i_flat = np.zeros(12)
    for leg in legs:
        p_i_flat[3 * leg.value : leg.value * 3 + 3] = p_i[leg]
        f_i_flat[3 * leg.value : leg.value * 3 + 3] = f_i[leg]

    X_k = np.hstack((Ori,p,omega,pdot))
    #print(X_k)
    U_k = np.hstack((p_i_flat, f_i_flat))

    return X_k, U_k

def solo_IK_np(p, R, p_i, elbow_up_front=True, elbow_up_hind=False):
    T_B = homog_np(p, R)
    rotate_90 = rot_mat_2d_np(np.pi / 2.0)
    q_i = {}
    for leg in legs:
        T_Bi = T_B @ B_T_Bi[leg]
        Bi_T = reverse_homog_np(T_Bi)
        Bi_p_i = mult_homog_point_np(Bi_T, p_i[leg])
        # assert abs(Bi_p_i[1]) < eps # foot should be in shoulder plane
        x_z = rotate_90 @ np.array([Bi_p_i[0], Bi_p_i[2]])
        if leg == legs.FL or leg == legs.FR:
            q1, q2 = planar_IK_np(L_thigh, L_calf, x_z[0], x_z[1], elbow_up_front)
        else:
            q1, q2 = planar_IK_np(L_thigh, L_calf, x_z[0], x_z[1], elbow_up_hind)
        q_i[leg] = np.array([q1, q2])

    return q_i




    