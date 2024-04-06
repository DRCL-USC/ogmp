
from preview_cmmp.utils import (
    legs,
    rot_mat_2d_np,
    homog_np,
    mult_homog_point_np,
    B_T_Bi,
    B_p_Bi,
    C_p_Bi,
    L,
    W,
    L_calf,
    L_thigh,
    extract_state_np,
    rot_from_rpy,
    solo_IK_np,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.plot_utils import plot_box
import matplotlib

# import seaborn as sns

# plt.style.use("seaborn")

# draws a coordinate system defined by the 4x4 homogeneous transformation matrix T
def draw_T(T):
    axis_len = 0.1
    origin = T[:3, 3]
    axis_colors = ["r", "g", "b"]
    for axis in range(3):
        
        axis_head = origin + axis_len * T[:3, axis]

        axis_coords = np.vstack((origin, axis_head)).T

        line = plt.plot([], [])[0]
        line.set_data(axis_coords[0], axis_coords[1])
        line.set_3d_properties(axis_coords[2])
        line.set_color(axis_colors[axis])


def draw(Ori, p, p_i, f_i,X,k,ax):
    
    R = rot_from_rpy(Ori[0],Ori[1],Ori[2]) 
    
    T_B = homog_np(p, R)
    p_Bi = {}
    c_Bi = {}
    for leg in legs:
        p_Bi[leg] = mult_homog_point_np(T_B, B_p_Bi[leg])
        c_Bi[leg] = mult_homog_point_np(T_B, C_p_Bi[leg])


    # draw body
    # body_coords = np.vstack(
    #     (p_Bi[legs.FL], p_Bi[legs.FR],c_Bi[legs.FR], c_Bi[legs.FL],p_Bi[legs.FL])
    # ).T
    
    # line = plt.plot([], [])[0]
    # line.set_data(body_coords[0], body_coords[1])
    # line.set_3d_properties(body_coords[2])
    # line.set_color("b")
    # line.set_marker("o")

    # inverse and forward kinematics to extract knee location
    # q_i = solo_IK_np(p, R, p_i, elbow_up_front, elbow_up_hind)
    # p_knee_i = {}
    # p_foot_i = {}
    # for leg in legs:
    #     Bi_xz_knee = rot_mat_2d_np(q_i[leg][0] - np.pi / 2.0) @ np.array([L_thigh, 0.0])
    #     Bi_xz_foot = Bi_xz_knee + rot_mat_2d_np(
    #         q_i[leg][0] - np.pi / 2.0 + q_i[leg][1]
    #     ) @ np.array([L_calf, 0.0])
    #     Bi_p_knee_i = np.array([Bi_xz_knee[0], 0.0, Bi_xz_knee[1]])
    #     Bi_p_foot_i = np.array([Bi_xz_foot[0], 0.0, Bi_xz_foot[1]])
    #     T_Bi = T_B @ B_T_Bi[leg]
    #     p_knee_i[leg] = mult_homog_point_np(T_Bi, Bi_p_knee_i)
    #     p_foot_i[leg] = mult_homog_point_np(T_Bi, Bi_p_foot_i)

    # ensure foot positions match the values calculated from IK and FK
    # note that the y position of the legs are allowed to deviate from 0 by
    # amount eps in the kinematics constraint, so we use something larger here
    # to check if the error is "not close to zero"
    # for leg in legs:
        
    #     assert np.linalg.norm(p_foot_i[leg] - p_i[leg]) < np.sqrt(eps)

    # draw legs
    # for leg in legs:
    #     leg_coords = np.vstack((p_Bi[leg], p_i[leg])).T
    #     # leg_coords = np.vstack((p_Bi[leg], p_i[leg])).T
    #     # leg_coords = np.atleast_2d(p_i[leg]).T
        
    #     line = plt.plot([], [])[0]
    #     # if p_i[leg][2]<0.01:
    #     line.set_data(leg_coords[0], leg_coords[1])
    #     line.set_3d_properties(leg_coords[2])
        
    #     # line.set_linewidth(0.01)
    #     line.set_color("g")
    #     line.set_marker("o")

    
    
    '''plotting box for the body'''
    plot_box(ax=ax, A2B=T_B, size=[0.11,0.194,0.247], color="g", alpha=0.5*(k/X.shape[1])+0.1,
         wireframe=False)


    ''' animate com traj'''
    line_1= plt.plot(X[3],X[4],X[5],lw=1,c='b')[0]
    line_1.set_data(X[3:5,:k])
    line_1.set_3d_properties(X[5,:k])
    line_1.set_color("black")

    draw_T(np.eye(4))
    draw_T(T_B)
    
    


def init_fig(X,g_b):
    anim_fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(anim_fig)
    anim_fig.add_axes(ax)
    x_start = X[3][0]-0.2
    x_end = X[3][-1]+0.2

    if X[3][0]>X[3][-1]-0.05:
        x_start-=0.3
        x_end +=0.3

    '''The axes are set'''
    ax.view_init(azim=-75,elev=5)
    ax.set_xlim3d([x_start, x_end])
    ax.set_ylim3d([-0.75, 0.75])
    ax.set_zlim3d([0, 2.0])
    # ax.set_axis_off()
    #ax.set_box_aspect([1, 1, 1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_zticks([])
    ax.w_xaxis.line.set_alpha(0.01)
    ax.w_yaxis.line.set_alpha(0.01)
    ax.w_zaxis.line.set_alpha(0.01)
    
    
    '''The terrain is set '''
    x_a = np.linspace(x_start, x_end, 100)
    y_a = np.linspace(0, 0.75, 100)
    X_mesh, Y_mesh = np.meshgrid(x_a, y_a)
    Z_mesh = np.zeros_like(X_mesh)  
    for i in range(len(g_b)):
        mask = np.logical_and(X_mesh >= g_b[i][0]+0.01, X_mesh <= g_b[i][1])
        Z_mesh[mask] = g_b[i][2]
    
    # Plot the flat plane
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.75,color='b')
    
   
    return anim_fig, ax

def animate_traj(X, U,g_b, dt, fname=None, display=True, repeat=True,motion_options={}):
    anim_fig, ax = init_fig(X,g_b)
    
    
    def draw_frame(k,X,ax):
        Ori, p, pdot, omega, p_i, f_i = extract_state_np(X, U, k)

        while ax.lines:
            ax.lines.pop()
        draw(Ori, p, p_i, f_i,X,k,ax)

    N = X.shape[1] - 1

    anim = animation.FuncAnimation(
        anim_fig,
        draw_frame,
        frames=np.arange(0, N+1, 10),
        fargs=(X,ax),
        interval=dt * 1000.0,
        repeat=repeat,
        blit=False,
    )

    if fname is not None:
        print("saving animation at ./" + fname + ".mp4...")
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(0.07 / dt), metadata=dict(artist="Me"), bitrate=1000)
        anim.save(fname + ".mp4", writer=writer)
        print("finished saving videos/" + fname + ".mp4")
        

    if display:
        plt.show()
        anim_fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0)
        # anim_fig.savefig("./results/gap.png")


def init_fig_rot(X,h_blk):
    anim_fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(anim_fig)
    anim_fig.add_axes(ax)
    x_start = X[3][0]
    x_end = X[3][-1]

    if x_start>x_end:
        x_start,x_end = x_end,x_start
        
    # print(x_start,x_end)

    '''The axes are set'''
    ax.view_init(azim=-30,elev=10)
    ax.set_xlim3d([x_start-0.2, x_end+0.2])
    ax.set_ylim3d([-0.75, 0.75])
    ax.set_zlim3d([0, 3.0])

    x_start = X[3][0]-0.2
    x_end = X[3][-1]+0.2

   
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_zticks([])
    ax.w_xaxis.line.set_alpha(0.01)
    ax.w_yaxis.line.set_alpha(0.01)
    ax.w_zaxis.line.set_alpha(0.01)
    
    
    '''The terrain is set '''
    x_a = np.linspace(x_start, x_end, 100)
    y_a = np.linspace(-0.25, 0.25, 100)
    X_mesh, Y_mesh = np.meshgrid(x_a, y_a)
    Z_mesh = np.zeros_like(X_mesh)  
    
    mask = np.logical_and(X_mesh >= -x_end, X_mesh <= -x_start)
    Z_mesh[mask] = h_blk
    # Plot the flat plane
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.75,color='b')
   
    return anim_fig, ax

def animate_traj_rot(X, U,h_blk, dt, fname=None, display=True, repeat=True,motion_options = {}):
    anim_fig, ax = init_fig_rot(X,h_blk)
    
    
    def draw_frame(k,X,ax):
        Ori, p, pdot, omega, p_i, f_i = extract_state_np(X, U, k)

        while ax.lines:
            ax.lines.pop()
        draw(Ori, p, p_i, f_i,X,k,ax)

    
    X_down = np.hstack((X[:,::5],X[:,-1].reshape(12,1)))
    N = X_down.shape[1] - 1

    anim = animation.FuncAnimation(
        anim_fig,
        draw_frame,
        frames=np.arange(0, N+1, 1),
        fargs=(X_down,ax),
        interval=dt * 1000.0,
        repeat=repeat,
        blit=False,
    )

    if fname is not None:
        print("saving animation at ./" + fname + ".mp4...")
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(0.07 / dt), metadata=dict(artist="Me"), bitrate=1000)
        anim.save(fname + ".mp4", writer=writer)
        print("finished saving videos/" + fname + ".mp4")
        

    if display:
        plt.show()
        anim_fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0)
        # anim_fig.savefig("./results/gap.png")
        


if __name__ == "__main__":
    from utils import rot_mat_np

    p = np.array([0.0, 0.0, 0.3])
    R = rot_mat_np(np.array([0, 1, 0]), 0.1)
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = B_p_Bi[leg]
        f_i[leg] = np.array([0.0, 0.0, 3.0])

    anim_fig, ax = init_fig()
    draw(p=p, R=R, p_i=p_i, f_i=f_i)
    plt.show()
