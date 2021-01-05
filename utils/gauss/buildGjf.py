import glob
import os
import pandas as pd
import numpy as np
import re
import shutil

def gaussian_gen(datadir, out_dir, phase):
    """ 
    Add header for guassian optimization  
    header_file: the file with Gaussian calculation header information
    """
    for file in glob.iglob(os.path.join(datadir, '*.com')):
        #print(file)
        name = file.split('/')[-1].split('.')[0]
        
        a = []
        temp = []
        with open(file) as f:
            for line in f.readlines():
                a.append(line)
        if a:      
            check_point='%Chk=' + '"' + name + '_' + phase + '"'
            cpu = '%NProcShared=2'
            mem = '%Mem=1GB'
            if phase == 'water':
               method = '#  B3LYP/6-31G* scrf=(smd, solvent=water)'
            if phase == 'octanol':
               method = '#  B3LYP/6-31G* scrf=(smd, solvent=n-octanol)'
            if phase == 'gas':
               method = '# opt freq B3LYP/6-31G*'
            space = ' '
            note = name + space + phase + space + 'phase'

            with open(out_dir + name + '_' + phase + '.gjf', 'w') as file:
                temp_claim = [check_point,cpu,mem,method,space,note,space]
                for item in temp_claim:
                    file.write('%s\n' % item)
                for line in a[5:]:
                    file.write(line)

phase='octanol'
out_path='/beegfs/dz1061/datasets/Frag20/Frag20_13/' + phase + '_smd_gjf/'
gaussian_gen('/beegfs/dz1061/datasets/Frag20/Frag20_13/gas_opt_com', out_path, phase)
