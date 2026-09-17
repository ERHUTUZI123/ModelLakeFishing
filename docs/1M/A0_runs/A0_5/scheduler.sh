set -eu
scontrol show job 1540848
scontrol show node watgpu308
scontrol show node watgpu608
squeue --start -j 1540848
sinfo -N -o '%N %t %m %G' | sort -u
