import numpy as np
import copy
from dtsd.envs.src.transformations import euler_to_rmat
import mujoco
import mujoco.viewer
# from perlin_noise import PerlinNoise
from dtsd.envs.src.misc_funcs import *




equality_constraint_id2type = {
                                'connect':0,
                                'weld':1,
                                'joint':2,
                                'tendon':3,
                                'distance':4
                                }
 
class mujoco_sim():
    """simulation base for all MuJoCo based environments."""
    def __init__(self, **kwargs ):
        
        self.sim_params = kwargs
        self.model = mujoco.MjModel.from_xml_path(self.sim_params['model_path'])
        self.data = mujoco.MjData(self.model)

        self.nominal_model = copy.deepcopy(self.model)
        
        # hfield params for terrain 
        try:
            self.hfield_x_index2len = 2 * self.model.hfield_size[0][0] / self.model.hfield_nrow[0]    
            self.hfield_y_index2len = 2 * self.model.hfield_size[0][1] / self.model.hfield_ncol[0]   
            self.hfield_scale = self.model.hfield_size[0][2]
        except:
            print("no height field in model")

        # viewer and viewer utils
        self.viewer_paused = True
        if self.sim_params['render']:
            self.viewer = mujoco.viewer.launch_passive(
                                                        self.model, 
                                                        self.data,
                                                        show_left_ui=False,
                                                        show_right_ui=False,
                                                        key_callback=self.viewer_key_callback,
                                                        )        

        else:
            self.viewer = None
        
        # visual markers
        self.vis_markers = []
        # manipulatable camera
        self.free_camera = mujoco.MjvCamera()

    # viewer related functions
    def viewer_key_callback(self,keycode):
        if chr(keycode) == ' ':
            # print('space')
            self.viewer_paused = not self.viewer_paused
        elif chr(keycode) == 'E':
            self.viewer.opt.frame = not self.viewer.opt.frame

    def init_renderers(self):
        self.renderer = mujoco.Renderer(self.model,height=1024, width=1440)

    def delete_renderers(self):
        del self.renderer

    def set_default_camera(self):
        mujoco.mjv_defaultCamera(self.free_camera)
        if self.sim_params['render']:
            mujoco.mjv_defaultCamera(self.viewer.cam)

    def get_frame_from_renderer(self,cam_name='viewer'):
        
        if cam_name == 'viewer':
            self.renderer.update_scene(
                                        self.data,
                                        camera=self.viewer.cam
                                        )
        elif cam_name == 'free_camera':
            self.renderer.update_scene(
                                        self.data,
                                        camera=self.free_camera
                                        )

        else:
            self.renderer.update_scene(
                                        self.data,
                                        camera=self.obj_name2id(name=cam_name,type='camera')
                                        )
            pass
        return self.renderer.render()
    
    def update_camera(
                        self,
                        cam_name ,
                        pos=[0.0, 0.0, 0.5],
                        azim = 90,
                        elev = -5,
                        dist = 2.0,
                        ):
        # self.viewer.cam.fixedcamid = 0
        # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        if cam_name == 'viewer':
            self.viewer.cam.distance = dist
            for i in range(3):        
                self.viewer.cam.lookat[i]= pos[i]
            self.viewer.cam.elevation = elev
            self.viewer.cam.azimuth = azim
        elif cam_name == 'free_camera':
            self.free_camera.distance = dist
            for i in range(3):
                self.free_camera.lookat[i]= pos[i]
            self.free_camera.elevation = elev
            self.free_camera.azimuth = azim
    
    def put_markers(self):
        s_i = self.viewer.user_scn.ngeom
        
        for marker_param in self.vis_markers:

            # set default values for markers
            if not exists_in('type',marker_param):
                marker_param['type'] = mujoco.mjtGeom.mjGEOM_BOX
            if not exists_in('mat',marker_param):
                marker_param['mat'] = np.eye(3).flatten()
            
            mujoco.mjv_initGeom(
                                self.viewer.user_scn.geoms[s_i],
                                **marker_param
                                )
            
            s_i += 1
        self.viewer.user_scn.ngeom = s_i

    def put_connector(self,
                      points,
                      line_width=0.0025,
                      rgba=[0.0,1,0.0,1],
                      ):

        s_i = self.viewer.user_scn.ngeom

        for i,p in enumerate(points[:-1]):

            mujoco.mjv_initGeom(
                            self.viewer.user_scn.geoms[s_i ],
                            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=[0.0, 0, 0],
                            pos=[0,0,0],
                            mat=np.zeros(9),
                            rgba=rgba,
                            )
            mujoco.mjv_makeConnector(
                                        self.viewer.user_scn.geoms[s_i],
                                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        width=line_width,
                                        a0=p[0],
                                        a1=p[1],
                                        a2=p[2],
                                        b0=points[i+1][0],
                                        b1=points[i+1][1],
                                        b2=points[i+1][2],
                                    )
            from_pt = str(p[0])+" "+str(p[1])+" "+str(p[2])
            to_pt = str(points[i+1][0])+" "+str(points[i+1][1])+" "+str(points[i+1][2])
            # print("<geom class=\"visual\" type=\"capsule\" size=\"0.0025\" fromto=\""+from_pt+" "+to_pt+"\" rgba=\"0.0 1.0 0.0 1.0\"/>")

            s_i += 1
  
        
        self.viewer.user_scn.ngeom = s_i

    def clear_markers(self):
        # remove all previous markers in scene
        self.vis_markers = []

    def clear_user_scn(self):
        self.viewer.user_scn.ngeom = 0
        self.clear_markers()

    # simulation related functions
    def reset(self): # -1 is random

        if self.sim_params['render']:
            self.clear_user_scn()
        mujoco.mj_resetData(self.model,self.data)        
        self.generate_terrain()

        ob = None
        # TBC
        # self.set_required_weld_contraints()

        return ob

    @property
    def dt(self):
        # print(self.model.opt.timestep)
        return self.model.opt.timestep 

    def set_control(self,ctrl):
        self.data.ctrl[:] = ctrl

    def simulate_n_steps(self, n_steps=1):        
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def forward(self):        
        mujoco.mj_forward(self.model, self.data)    

    def close(self):
        if self.sim_params['render']:
            self.viewer.close()

    # utility functions
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_xpos(self,body_name):
        body_id = self.obj_name2id(name=body_name,type='body')

        return self.data.xpos[body_id]

    def get_body_xquat(self,body_name):
        body_id = self.obj_name2id(name=body_name,type='body')

        return self.data.xquat[body_id]


    def get_terrain_height_at(self,pos):
        i_current = int(pos[0] /self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_current = int(pos[1] /self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])
        return self.hfield_scale*self.model.hfield_data[j_current*self.model.hfield_ncol[0] + i_current]

    def get_terrain_infront(self,pos,halfwidth_y,halfwidth_x):

        i_start = int( (pos[0]-halfwidth_x) /self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        i_end   = int( (pos[0]+halfwidth_x) /self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        
        j_start = int( (pos[1]-halfwidth_y) /self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])
        j_end = int( (pos[1]+halfwidth_y) /self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])
        
        terrain_map = np.zeros((i_end-i_start,j_end-j_start))

        for i in range(i_start,i_end):
            for j in range(j_start,j_end):
                terrain_map[i-i_start,j-j_start] = self.hfield_scale*self.model.hfield_data[j*self.model.hfield_ncol[0] + i]

        return terrain_map

    def get_terrain_height_at_xy(self,xy):
        i_current = int(xy[0] /self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_current = int(xy[1] /self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])
        return self.hfield_scale*self.model.hfield_data[j_current*self.model.hfield_ncol[0] + i_current]
        
    def get_sensordata(self,name):
        

        sensor_id = self.obj_name2id(name,type='sensor')
        start_n = self.model.sensor_adr[sensor_id]
        

        if sensor_id == self.model.nsensor -1:
            return self.data.sensordata[start_n:]
        else:
            end_n = self.model.sensor_adr[sensor_id+1]
  
            return self.data.sensordata[start_n:end_n]

    def call_mujoco_func(   
                        self,
                        function_name,
                        send_model_and_data=True
                        ):
        if send_model_and_data:
            getattr(mujoco, function_name)(self.model,self.data)
        else:
            getattr(mujoco, function_name)()

    def set_vclone_color(self,set='switch'):

        geoms_body_ids = []        
        for geom_id,body_id in enumerate(self.model.geom_bodyid):    
            body_name = self.obj_id2name(obj_id=body_id,type='body')
            if 'v_' in body_name and 'joint' not in body_name:
                geoms_body_ids.append(geom_id)

        
        if set=='switch':
            if self.model.geom_rgba[geoms_body_ids[0]][1] == 0.5: # current color green
                color2set = [0.5, 0.0, 0.0,0.5]
            else: # current color red
                color2set = [0.0, 0.5, 0.0,0.5]
        elif set=='green':
            color2set = [0.0, 0.5, 0.0,0.5]          
        elif set=='red':
            color2set = [0.5, 0.0, 0.0,0.5]
        else:
            print("requested color not registered here, ignored call")
            return

        for geom_id in geoms_body_ids:        
            self.model.geom_rgba[geom_id] = color2set

    def set_foot_terrain_solref(self,val=None,fraction_nominal=1):

        terrain_id = self.obj_name2id(name='terrain',type='geom')

        # print('nominal:',self.nominal_model.geom_solref[terrain_id,:])


        if val is None:
            val = fraction_nominal*self.nominal_model.geom_solref[terrain_id,:]
        # print('updated:',val)

        self.model.geom_solref[terrain_id,:] = val    

        toe_ids = [self.obj_name2id(name=body_name,type='body') for body_name in ['R_toe','L_toe']]  
        
        for geom_id,bodyi_id in enumerate(self.model.geom_bodyid):
            if bodyi_id in toe_ids:
                self.model.geom_solref[geom_id,:] = val 

    def set_foot_terrain_friction(self,val=None,fraction_nominal=1):

        terrain_id = self.obj_name2id(name='terrain',type='geom')


        if val is None:
            val = fraction_nominal*self.nominal_model.geom_friction[terrain_id,0]
        self.model.geom_friction[terrain_id,0] = val # sliding friction only 

        
        toe_ids = [self.obj_name2id(name=body_name,type='body') for body_name in ['R_toe','L_toe']]  
        
        for geom_id,bodyi_id in enumerate(self.model.geom_bodyid):
            

            if bodyi_id in toe_ids:
                self.model.geom_friction[geom_id,0] = val # sliding friction only

    def set_body_friction(self,body,val=None,fraction_nominal=1):

        if body == 'terrain':
            terrain_id = self.obj_name2id(name='terrain',type='geom')
        
            if val is None:
                val = fraction_nominal*self.nominal_model.geom_friction[terrain_id,0]
            
            self.model.geom_friction[terrain_id,0] = val # sliding friction only 

        else:
            body_id = self.obj_name2id(name=body,type='body')
            for geom_id,bodyi_id in enumerate(self.model.geom_bodyid):
                if bodyi_id == body_id:
 
                    if val is None:
                        val = fraction_nominal*self.nominal_model.geom_friction[geom_id,0] 
                        # print('nominal:',self.nominal_model.geom_friction[geom_id,0],'val:',val)
 
                    self.model.geom_friction[geom_id,0] = val # sliding friction only 
        
    def set_body_mass(self,body,val=None,fraction_nominal=1):

        body_id = self.obj_name2id(name=body,type='body')

        if val is None:
            val = fraction_nominal*self.nominal_model.body_mass[body_id]

        self.model.body_mass[body_id] = val

    def set_body_ipos(self,body,val):
        body_id = self.obj_name2id(name=body,type='body')
        self.model.body_ipos[body_id] = val

    def set_dof_damping(self,body,damping):
        body_id = self.obj_name2id(name=body,type='body')
        self.model.dof_damping[body_id] = damping

    def set_eq_contraints(self,eq_active={}):

        self.model.eq_active[:] = 0 # deactivate all 
        for eq_name in eq_active.keys():
            eq_id = self.obj_name2id(name=eq_name,type='equality')
            # activate eq constraint
            self.model.eq_active[eq_id] = 1
            if 'data' in eq_active[eq_name].keys():
            	# temp hack, weied issue in resmotes system only
                if self.model.eq_data[eq_id].shape[0] > 7:
                    self.model.eq_data[eq_id][3] = eq_active[eq_name]['data'][0]
                    self.model.eq_data[eq_id][4] = eq_active[eq_name]['data'][1]
                    self.model.eq_data[eq_id][5] = eq_active[eq_name]['data'][2]
                    self.model.eq_data[eq_id][6] = eq_active[eq_name]['data'][3]
                    self.model.eq_data[eq_id][7] = eq_active[eq_name]['data'][4]
                    self.model.eq_data[eq_id][8] = eq_active[eq_name]['data'][5]
                    self.model.eq_data[eq_id][9] = eq_active[eq_name]['data'][6]
                else:
                    self.model.eq_data[eq_id] = eq_active[eq_name]['data']
            if 'body1' in eq_active[eq_name].keys():
                self.model.eq_obj1id[eq_id] = self.obj_name2id(name=eq_active[eq_name]['body1'],type='body')
            if 'body2' in eq_active[eq_name].keys():
                self.model.eq_obj2id[eq_id] = self.obj_name2id(name=eq_active[eq_name]['body2'],type='body')

    def remove_all_weld_contraints(self):        
        if self.model.eq_type != None:
            for eq_id,eq_type in enumerate(self.model.eq_type):
                if eq_type ==  equality_constraint_id2type['weld']:
                    self.model.eq_active[eq_id] = 0

    def obj_name2id(self,name,type='body'):
        type = type.upper()
        return mujoco.mj_name2id(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    name
                                )

    def obj_id2name(self,obj_id,type='body'):
        type = type.upper() 
        return mujoco.mj_id2name(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    obj_id
                                )

    def contact_bw_bodies(
                            self,
                            body1,
                            body2,
                          ):

        geoms_body1_ids = []        
        if body1 == 'terrain':
            geoms_body1_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body1_id = self.obj_name2id(name=body1,type='body')
            
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body1_id:
                    geoms_body1_ids.append(geom_id)

        geoms_body2_ids = []
        if body2 == 'terrain':
            geoms_body2_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body2_id = self.obj_name2id(name=body2,type='body')
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body2_id:
                    geoms_body2_ids.append(geom_id)
        


        # print('body1_geoms:',geoms_body1_ids,'body2_geoms:',geoms_body2_ids)
        for n in range(self.data.ncon):
            contact = self.data.contact[n]
            if (contact.geom1 in geoms_body1_ids and contact.geom2 in geoms_body2_ids) or \
               (contact.geom2 in geoms_body1_ids and contact.geom1 in geoms_body2_ids) :
                # to make the function name-order agnostic

                return True,contact
                

        return False,None
    def total_contact_force_bw_bodies(
                            self,
                            body1,
                            body2,
                            ):
        # collect all geom for body1 and body2
        geoms_body1_ids = []        
        if body1 == 'terrain':
            geoms_body1_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body1_id = self.obj_name2id(name=body1,type='body')
            
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body1_id:
                    geoms_body1_ids.append(geom_id)
        

        geoms_body2_ids = []
        if body2 == 'terrain':
            geoms_body2_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body2_id = self.obj_name2id(name=body2,type='body')
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body2_id:
                    geoms_body2_ids.append(geom_id)
        
        n_contact_points = 0
        total_contact_force = np.zeros(6)
        for n in range(self.data.ncon):
            contact = self.data.contact[n]
            if (contact.geom1 in geoms_body1_ids and contact.geom2 in geoms_body2_ids) or \
               (contact.geom2 in geoms_body1_ids and contact.geom1 in geoms_body2_ids) :
                contact_frc_at_point = np.zeros(6)
                mujoco.mj_contactForce(self.model,self.data,n,contact_frc_at_point)
                total_contact_force += contact_frc_at_point
                n_contact_points += 1

        return total_contact_force

    def all_contact_bw_bodies(
                            self,
                            body1,
                            body2,
                          ):

        geoms_body1_ids = []        
        if body1 == 'terrain':
            geoms_body1_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body1_id = self.obj_name2id(name=body1,type='body')
            
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body1_id:
                    geoms_body1_ids.append(geom_id)

        geoms_body2_ids = []
        if body2 == 'terrain':
            geoms_body2_ids.append(self.obj_name2id(name='terrain',type='geom'))
        else:
            body2_id = self.obj_name2id(name=body2,type='body')
            for geom_id,body_id in enumerate(self.model.geom_bodyid):
                if body_id == body2_id:
                    geoms_body2_ids.append(geom_id)
        

        total_contcts = 0
        total_cfrc = np.zeros((6,1))
        # print('body1_geoms:',geoms_body1_ids,'body2_geoms:',geoms_body2_ids)
        for n in range(self.data.ncon):
            contact = self.data.contact[n]
            if (contact.geom1 in geoms_body1_ids and contact.geom2 in geoms_body2_ids) or \
               (contact.geom2 in geoms_body1_ids and contact.geom1 in geoms_body2_ids) :
                total_contcts += 1
                cfrc = np.zeros((6,1)) 
                mujoco.mj_contactForce(self.model,self.data,n,cfrc)
                total_cfrc += cfrc
        
        return total_contcts,total_cfrc

    # debugger print functions
    def print_all_contacts(self):


        for coni in range(self.data.ncon):
            print('  Contact %d:' % (coni,))
            con = self.data.contact[coni]
            print('    dist     = %0.3f' % (con.dist,))
            print('    pos      = %s' % (con.pos),)
            print('    frame    = %s' % (con.frame),)
            print('    friction = %s' % (con.friction),)
            print('    dim      = %d' % (con.dim,))
            print('    geom1    = %d' % (con.geom1,))
            # print('    g1_name  = %s' % ( self.model.geom_id2name(con.geom1),))
            
            print('    geom2    = %d' % (con.geom2,))
            # print('    g2_name  = %s' % ( self.model.geom_id2name(con.geom2),))

    def print_all_eq_contraints(self):
        
        for i,(active,data,type,obj1,obj2) in enumerate(zip(
                                    self.model.eq_active,
                                    self.model.eq_data,
                                    self.model.eq_type,
                                    self.model.eq_obj1id,
                                    self.model.eq_obj2id,
                                    )):
            print('eq:',self.obj_id2name(obj_id=i,type='equality'),
                  'active:',active,'data:',data,'type:',type,'obj1:',obj1,'obj2:',obj2)

    def set_required_weld_contraints(self):
        # TBC
        for eq_id,(eq_obj1id,eq_obj2id,eq_type) in enumerate(zip(self.model.eq_obj1id,self.model.eq_obj2id,self.model.eq_type)):
            
            if eq_type ==  equality_constraint_id2type['weld']:
                
                if self.model.body_id2name(eq_obj2id) == 'world':
                    if self.sim_params['set_on_rack']:
                        self.model.eq_active[eq_id] = 1
                
                if 'mocap_' in self.model.body_id2name(eq_obj1id) or 'mocap_' in self.model.body_id2name(eq_obj2id):
                    if self.sim_params['mocap']:
                        self.model.eq_active[eq_id] = 1 

    # heightmap terrain related functions
    def generate_terrain(self):

        if 'terrain' not in self.sim_params.keys():
            # plane
            self.generate_terrain_plane()
            

        else:

            # sample a random isntance of each of the terrain, in te espective paramter ranges
            terrain_param_random_sample = {} 
            for terrain_name in self.sim_params['terrain'].keys():
                terrain_param_random_sample.update({terrain_name:{}})
                for param_name in self.sim_params['terrain'][terrain_name].keys():
                    param_val = np.random.uniform(
                                                    low=self.sim_params['terrain'][terrain_name][param_name][0], 
                                                    high=self.sim_params['terrain'][terrain_name][param_name][-1] 
                                                    )
                    terrain_param_random_sample[terrain_name].update({param_name:param_val})
            
            # generate terrains, preferably non overlapping
            for terrain_name in self.sim_params['terrain'].keys():
                
                if '_instance' in terrain_name:
                    terain_type_name = terrain_name.split("_instance")[0]
                else:
                    terain_type_name = terrain_name
                
                gen_terrain_func = getattr(self, 'generate_terrain_'+terain_type_name)
                gen_terrain_func(**terrain_param_random_sample[terrain_name])

    def generate_terrain_plane(self):
        self.model.hfield_data[:] = 0

    def generate_terrain_upstairs(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                step_run_lim = 0.25,
                                step_rise_lim = 0.05,
                                ):    
        
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        # flush before +=
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = 0


        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] += step_rise_lim* int((x_i - x_start_lim) / step_run_lim)

    def generate_terrain_hills(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                amplitude_lim = 0.25,
                                octaves_lim = 0.05,
                                ):
        noise = PerlinNoise(octaves_lim)
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])


        # flush before +=
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = 0

        
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] += amplitude_lim*noise([i/self.model.hfield_nrow[0], j/self.model.hfield_ncol[0]])

    def generate_terrain_plateau(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                height_lim = 4
                                ):
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])


        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = height_lim / self.hfield_scale

    def generate_terrain_plateau2(
                                self,
                                x_start_lim = 0.5,
                                x_len_lim = 2,
                                y_start_lim = -2,
                                y_len_lim = 2,
                                height_lim = 4
                                ):
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int( (x_start_lim+x_len_lim)/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int( (y_start_lim+y_len_lim)/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = height_lim

    def generate_terrain_sinusoidal(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                amplitude_lim = 0.25,
                                ang_freq_lim = 0.05,
                                phase_offset_lim = np.pi/4

                                ):
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = amplitude_lim*np.sin( ang_freq_lim*(x_i-x_start_lim) + phase_offset_lim)

    def generate_terrain_downstairs(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                step_run_lim = 0.25,
                                step_rise_lim = 0.05,
                                ):    
        
        
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])


        # flush before +=
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = 0


        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis 
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] += step_rise_lim* int((x_end_lim - x_i ) / step_run_lim)

    def generate_terrain_upstairs_plateau_downstairs(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                step_run_lim = 0.25,
                                step_rise_lim = 0.05,
                                fraction_len_plateau_lim = 0.333, 
                                ):    
        
        
        # to make it an integral multiple of the hfield's x presision 
        step_run_lim = int(step_run_lim/self.hfield_x_index2len)*self.hfield_x_index2len 
        
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        
        # up and down stairs are off equal fraction, i.e. the remainin of pleatu region
        total_x_len = (x_end_lim - x_start_lim)
        fraction_len_stairs = 0.5*(1. - fraction_len_plateau_lim)        
        plateau_end = 0        
        plateau_height = self.model.hfield_data[j_start*self.model.hfield_ncol[0] + i_start]
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis 
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                x_len_covered = x_i - x_start_lim
                if x_len_covered < fraction_len_stairs*total_x_len:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = step_rise_lim* int((x_i-x_start_lim ) / step_run_lim)
                    plateau_height = self.model.hfield_data[j*self.model.hfield_ncol[0] + i]
                elif fraction_len_stairs*total_x_len < x_len_covered and x_len_covered < (fraction_len_stairs+fraction_len_plateau_lim)*total_x_len:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = plateau_height
                    plateau_end = x_i
                elif (fraction_len_stairs+fraction_len_plateau_lim)*total_x_len < x_len_covered and x_len_covered < total_x_len:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = plateau_height - step_rise_lim* int((x_i-plateau_end ) / step_run_lim)

    def generate_terrain_plateau_slope(
                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                elevation_lim = 5,
                                fraction_len_plateau_lim = 0.5, 
                                ):    
        
        
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        # integral multiple of terrain precision
        x_end_lim = int(x_end_lim/self.hfield_x_index2len)*self.hfield_x_index2len 
        x_start_lim = int(x_start_lim/self.hfield_x_index2len)*self.hfield_x_index2len 

        total_x_len = (x_end_lim - x_start_lim)
        fraction_len_slope = 1. - fraction_len_plateau_lim        
        i_h = np.abs(fraction_len_slope*total_x_len*np.tan(np.radians(elevation_lim)))
        plateau_end = x_start_lim
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis 
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                if (x_i - x_start_lim) < fraction_len_plateau_lim*total_x_len:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = i_h
                    plateau_end = x_i
                else:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = i_h +  (x_i - plateau_end)*np.tan(np.radians(elevation_lim))

    def generate_terrain_slope(

                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                elevation_lim = 5,

                                ):
        # assumes the lowest height is alays zero
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])



        slope_length = x_end_lim - x_start_lim
        
        if elevation_lim >= 0:
            i_h = 0 
        else:
            i_h = np.abs(slope_length*np.tan(np.radians(elevation_lim)))

        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len  
                self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = i_h +  (x_i - x_start_lim)*np.tan(np.radians(elevation_lim))    
    
    def generate_terrain_slope_rotated(

                                self,
                                x_orig_lim = 0.5,
                                y_orig_lim = 0,
                                terrain_l_lim = 0.5,
                                terrain_w_lim = 0.5,
                                elevation_lim = 15,
                                about_z_rotate_lim = 0
                                ):
        

        # # assumes the lowest height is alays zero


        rmat_w2b = euler_to_rmat(np.radians([0,0,-45]))
        
        b_pos = [x_orig_lim,y_orig_lim,0]

        x_start_lim = x_orig_lim -0.5*terrain_l_lim
        x_end_lim = x_orig_lim +0.5*terrain_l_lim
        y_start_lim = y_orig_lim -0.5*terrain_w_lim
        y_end_lim = y_orig_lim +0.5*terrain_w_lim


        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        print(i_start,i_end)
        print(j_start,j_end)


        slope_length = x_end_lim - x_start_lim
        
        if elevation_lim >= 0:
            i_h = 0 
        else:
            i_h = np.abs(slope_length*np.tan(np.radians(elevation_lim)))

        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis
                
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len  
                y_j = ( j - 0.5*self.model.hfield_ncol[0] ) * self.hfield_y_index2len  
                h_ij = i_h +  (x_i - x_start_lim)*np.tan(np.radians(elevation_lim))  
                
                pt_xy_rot = b_pos + np.dot(rmat_w2b,np.array([x_i,y_j,0]))

                i_rot = int(pt_xy_rot[0]/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
                j_rot = int(pt_xy_rot[1]/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
                
                self.model.hfield_data[j_rot*self.model.hfield_ncol[0] + i_rot] = h_ij

    def generate_terrain_plateau_expslope(

                                self,
                                x_start_lim = 0.5,
                                x_end_lim = 2,
                                y_start_lim = -2,
                                y_end_lim = 2,
                                exp_h_lim = 5,
                                exp_k_lim = 5,
                                fraction_len_plateau_lim = 0.5,

                                ):
        # assumes the lowest height is alays zero
        i_start = int(x_start_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_start = int(y_start_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        i_end = int(x_end_lim/self.hfield_x_index2len + 0.5*self.model.hfield_nrow[0])
        j_end = int(y_end_lim/self.hfield_y_index2len + 0.5*self.model.hfield_ncol[0])

        # integral multiple of terrain precision
        x_end_lim = int(x_end_lim/self.hfield_x_index2len)*self.hfield_x_index2len 
        x_start_lim = int(x_start_lim/self.hfield_x_index2len)*self.hfield_x_index2len 
        

        # integral multiple of terrain precision
        x_end_lim = int(x_end_lim/self.hfield_x_index2len)*self.hfield_x_index2len 
        x_start_lim = int(x_start_lim/self.hfield_x_index2len)*self.hfield_x_index2len 


        total_x_len = (x_end_lim - x_start_lim)

        plateau_end = x_start_lim
        for i in range(i_start,i_end): # x axis
            for j in range(j_start,j_end): # y axis 
                x_i = ( i - 0.5*self.model.hfield_nrow[0] ) * self.hfield_x_index2len
                if (x_i - x_start_lim) < fraction_len_plateau_lim*total_x_len:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = exp_h_lim
                    plateau_end = x_i
                else:
                    self.model.hfield_data[j*self.model.hfield_ncol[0] + i] = exp_h_lim*np.exp(-exp_k_lim* (x_i - plateau_end)**2)
