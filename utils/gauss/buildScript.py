import glob
import os
import pandas as pd
import numpy as np
import re
import shutil


def gen_in(path_in, path_out, path_result, num_files):
        fileList = glob.glob(os.path.join(path_in, '*gjf'))
        print(len(fileList))
        num = int(len(fileList) / num_files)
        for k in range(num+1):
            print((k+1)*num_files)
            with open(path_out + str(k) + '_'+str(num_files)+'.in','w+') as f:
            #for file in os.listdir(path_in): 
                    #if file.endswith('.gjf'):
                        #with open(path_out + file[:-4] + '.in','w+') as f:
                                line1 = '#!/bin/bash'
                                line2 = '#'
                                line3 = '#SBATCH --job-name=' + str(k)
                                line4 = '#SBATCH --nodes=1'
                                line5 = '#SBATCH --cpus-per-task=2'
                                line6 = '#SBATCH --time=30:00:00'
                                line7 = '#SBATCH --mem=1GB'
                                linespace = ' '
                                line8 = 'module purge'
                                line9 = 'module load gaussian/intel/g16a03'
                                line10 = 'cd' + linespace + path_in
                                lines = [line1,line2,line3,line4,line5,line6,line7,linespace,linespace,line8,line9,linespace,line10]
                                for line in lines:
                                    f.write("%s\n" % line)
                                for file in fileList[k*num_files:(k+1)*num_files]:
                                    name = file.split('/')[-1].split('.')[0]
                                    line11='srun run-gaussian ' + file  + ' > ' + path_result + name + '_smd.log' + ' 2>&1' ' && mv ' + file + ' ' + path_in+'_finished;'
                                    f.write("%s\n" % line11)

phase='octanol'
gen_in('/beegfs/dz1061/datasets/Frag20/Frag20_13/'+phase+'_smd_gjf', '/beegfs/dz1061/datasets/Frag20/Frag20_13/'+phase+'_smd_in/', '/beegfs/dz1061/datasets/Frag20/Frag20_13/'+phase+'_smd_out/', num_files=200)
