import sys
sys.path.append("./")
from dtsd.envs.sim.mujoco_sim_base import mujoco_sim
import math
import matplotlib.pyplot as plt



sim = mujoco_sim( 
                  render=True,

                  model_path= 'dtsd/envs/rsc/models/mini_biped/xmls/biped_mesh_vis_simple_col.xml',
                #   model_path= 'dtsd/envs/rsc/models/mini_biped/xmls/biped_simple.xml',

                  )



if sim.sim_params['render']:
    sim.viewer._render_every_frame = False
    sim.viewer._paused = True
    sim.viewer.cam.distance = 2.25
    cam_pos = [sim.data.qpos[0], 0.0, 0.5]

    for i in range(3):        
        sim.viewer.cam.lookat[i]= cam_pos[i] 
    
    sim.viewer.cam.elevation = -5
    sim.viewer.cam.azimuth = 90


total_mass = 0
for body_id in range(sim.model.nbody):
    if 'v_' not in sim.obj_id2name(body_id):
      body_name = sim.obj_id2name(body_id)
      body_mass = sim.model.body_mass[body_id]
      body_inertia = sim.model.body_inertia[body_id]

      print(body_name, body_mass, body_inertia)
      total_mass += sim.model.body_mass[body_id]

print("total mass:", total_mass)


while True:
    # sim.simulate_n_steps()
    sim.viewer.sync()
# for i in range(sim.model.njnt):


#   # if sim.model.jnt_type[i] != 0:
#   if i >1 :
  
#     print('joint',i,
#             sim.model.jnt_range[i],
#             sim.model.jnt_dofadr[i],
#             sim.model.jnt_type[i],)
#     jpos = sim.model.jnt_range[i][0]

#     while jpos < sim.model.jnt_range[i][-1]:
#       jpos += math.radians(0.025)
      
#       sim.data.qpos[7:] = 0.0

#       sim.data.qpos[sim.model.jnt_dofadr[i]+1] = jpos

#       sim.simulate_n_steps()
#       if sim.sim_params['render']:
#         sim.render()      

# while True:
