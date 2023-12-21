#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a10:4
#SBATCH -o ./SLURM.%N.%j.out         # STDOUT
#SBATCH -e ./SLURM.%N.%j.err         # STDERR

# set container
gate_node='gate1'
image_path='/home2/labhosik/llama/fin.sqsh'

# parse image, container name
image_name=${image_path##*/}
container_name=${image_name%%.*}

# check old container in calculatation node and remove.
container_path="/scratch/enroot/$UID/data/$container_name"
test -d $container_path && 
    rm -rf $container_path

# connect gate:$gate_port and node:$node_port
user=`whoami`
echo $user@$gate_node
gate_port=`ssh $user@$gate_node "ruby -e 'require \"socket\"; puts Addrinfo.tcp(\"\", 0).bind {|s| s.local_address.ip_port }'"`
gate_port=`echo $gate_port | awk '{print \$1;}'`
node_port=`ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }'`
ssh $user@$gate_node -R $gate_port:localhost:$node_port -fN "while sleep 100; do; done"&

# print job info
echo "start at:" `date`
echo "node: $HOSTNAME"
echo "gate: $gate_node"
echo "container_name: $container_name"
echo "node_port: $node_port"
echo "*********************"
echo "gate_port: $gate_port    <-- pleaze copy this!!"
echo "*********************"
echo "jobid: $SLURM_JOB_ID"

enroot create -n $container_name $image_path
echo 'start enroot container'

mkdir -p ~/project
enroot start \
    --root \
    --rw \
    -m $HOME:/data \
    $container_name \
    -c "cd .."\
    -c 'sh ./data/llama/alpaca/fc.sh'
