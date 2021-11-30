#!/bin/bash

args=''
for i in "$@"; do 
    i="${i//\\/\\\\}"
    args="$args \"${i//\"/\\\"}\""
done

if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

/share/apps/singularity/bin/singularity \
    exec $nv \
    --overlay /scratch/dz1061/overlay/torch.ext3:ro \
    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
    /bin/bash -c "
if [ -f /ext3/torch.sh ]; then source /ext3/torch.sh; fi
eval $args
exit
"








