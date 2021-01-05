import glob
import os
import pandas as pd
import numpy as np
import re
import shutil

Frag_number=13
for file in glob.iglob('/beegfs/dz1061/datasets/Frag20/Frag20_'+str(Frag_number)+'/gas_opt_log/*log'):
    #file = files_use[idx]
    name = file.split('/')[-1].split('.')[0]
    cmd = '/share/apps/openbabel/3.0.0/intel/bin/obabel -ig09 ' + file + ' -ocom -O ' + '/beegfs/dz1061/datasets/Frag20/Frag20_'+str(Frag_number)+'/gas_opt_com/' + name + '_gas_opt.com'
    os.system(cmd)
    
    #break
