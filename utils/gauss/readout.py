import glob
import os
import pandas as pd
import numpy as np
import re
import shutil

def getGaussianLog(fn):
        """ 
        Get the information from Gaussian 09 output xx.log
        
        fn: str, gaussian output log
        Output:
            (str, [str * 16])
            (filename, [property list])
        If it is not finished, all values will be None
        
        Properties list contains 16 values. 
        The first 15 is the same as in QM9 and the last value is E(B3LYP) (Ha unit)
        
        """
        out = [None for i in range(1)]
        if not os.path.exists(fn):
            return fn, out
        with open(fn, 'r') as f:
            data = f.read()
            mo = re.search('Normal termination of Gaussian 09 at', data)
            if not mo:
                #print('Error: ', fn, ' does not finish!')
                return fn, out
        log = data.split('\n')
        
        proplist = ['E', 'Time']
        propdict = {i:None for i in proplist}
        #charges = {}
        for idx, item in enumerate(log):
            
            if item.startswith(' SCF Done'):
                #print(item)
                propdict['E'] = item.split()[4]
            if item.startswith(' Job cpu time'):
                #print(item)
                propdict['Time'] = int(item.split()[7])*60 + float(item.split()[9])
                #print(item.split()[7])
                
        #propdict.update(charges)
        #print(propdict)
        propvals = []
        for i in proplist:
            if propdict[i] != None:
                propvals.append(propdict[i])
            else:
                propvals.append(None)
        return fn, propvals

df = []
folder = 16
phase = 'gas'
for file in glob.iglob('/beegfs/dz1061/datasets/Frag20/Frag20'+'_'+str(folder)+'/'+phase+'_out_log/*.log'):
    name = file.split('/')[-1].split('.')[0].split('_')[0]
    _, res = getGaussianLog(file)
    res.append(int(name))
    df.append(res)
    #break
np.save('/beegfs/dz1061/datasets/Frag20/Frag20'+ '_'+str(folder)+'/'+phase+'_opt_log.npy', df)
