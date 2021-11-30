#!/bin/bash

args=''
for i in "$@"; do 
    i="${i//\\/\\\\}"
    args="$args \"${i//\"/\\\"}\""
done

if [ "$args" == "" ]; then args="/bin/bash"; fi

/share/apps/singularity/bin/singularity \
    exec \
    --overlay //scratch/dz1061/overlay/pytorch1.8.1-rocm4.0.1.sqf:ro \
    /scratch/work/public/hudson/images/rocm-4.0.1.sif \
    /bin/bash -c "
if [ -f /ext3/env.sh ]; then source /ext3/env.sh; fi
eval $args
exit
"








