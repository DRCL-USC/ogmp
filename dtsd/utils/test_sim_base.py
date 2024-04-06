import sys
sys.path.append("./")
import numpy as np
from dtsd.envs.sim.mujoco_sim_base import mujoco_sim
import time

sim = mujoco_sim( 
                  render=True,
                  model_path= 'dtsd/envs/rsc/models/hector_biped_v1/xmls/svsc_dd_box.xml',
                  )


total_mass = 0
for body_id in range(sim.model.nbody):
    if 'v_' not in sim.obj_id2name(body_id):
      print(sim.obj_id2name(body_id),sim.model.body_mass[body_id])
      total_mass += sim.model.body_mass[body_id]
print("total mass:", total_mass)

# joint names with id
print("qpos0:",sim.data.qpos)  
sim.reset()

for ji in range(sim.model.njnt):
  jname = sim.obj_id2name(obj_id=ji,type='joint')
  qi = 7-1+ji
  print(ji, qi, jname, sim.data.qpos[qi].round(3))

sim.viewer.cam.lookat[2] = 3.0

steps = 0
maxsteps = 10000
while True:
    if not sim.viewer_paused:
      sim.data.ctrl[3] = -10
      sim.simulate_n_steps()
      time.sleep(0.005)
      if sim.sim_params['render']:
        sim.viewer.sync()
      steps+=1
      if steps > maxsteps:
          break

