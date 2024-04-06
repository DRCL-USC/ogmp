import numpy as np
import os
import yaml
from .misc_funcs import exists_and_true

class logger_dummy: 
    def __init__(self,logger_conf):
        pass
    def update(self, env,sim_data):
        return
    def reset(self):
        return
    def export(self,export_name='dummy'):
        return
    
class logger:
    def __init__(
                    self, 
                    logger_conf,
                ):
        

        self.conf = logger_conf
        # os.makedirs(self.conf['export_path'],exist_ok=True)

        # for log_name in self.conf.keys():
            
        
        self.export_logs = {}
        for log_name in self.conf.keys():
            if log_name not in ['export_path','export_date_time','avoid_logfolder_names','export_conf']:

                self.conf[log_name] ={
                                                'source': self.conf[log_name],  
                                                'data_name': log_name
                                            }            
                
                if self.conf[log_name]['source'] == 'env':
                    self.conf[log_name]['data_name'] = 'curr_'+log_name
                
                self.export_logs.update({log_name:[]})

        self._env_exp_conf = None
        self.folderpath = None

    def update(
                self, 
                env,
                sim_data,
                ):

        for log_name in self.export_logs.keys():
            
            data = getattr(
                            eval(self.conf[log_name]['source']),
                            self.conf[log_name]['data_name'] 
                            )
            data = data.tolist() if isinstance(data,np.ndarray) else data
            
            self.export_logs[log_name].append(
                                                data
                                             )

            self._env_exp_conf = env.exp_conf

    def reset(self):
        for log_name in self.export_logs.keys():
            self.export_logs[log_name] = []
    
    def export(
                self,
                export_name='dummy',
                verbose=False,
                ):

        if verbose:
            self.print_log_status()        

        if exists_and_true('avoid_logfolder_names',self.conf):
            folderpath = self.conf['export_path']+'/'
        else:
            folderpath = self.conf['export_path']+'/'+self.conf['export_date_time']+'/'+export_name+'/'

        os.makedirs(folderpath,exist_ok=True)
        self.folderpath = folderpath
        np.savez_compressed(
                                folderpath+'/log.npz',
                                **self.export_logs
                            )

        if exists_and_true('export_conf',self.conf):
            tstng_exp_conf_file =  open(folderpath+'/exp_conf.yaml','w')
            yaml.dump(
                        self._env_exp_conf,
                        tstng_exp_conf_file,
                        default_flow_style=False,
                        sort_keys=False
                    )        


    def print_log_status(self):
        for log_name in self.export_logs.keys():
            print(
                    '\t',
                    log_name,
                    ':',
                    np.array( self.export_logs[log_name]).shape
                )