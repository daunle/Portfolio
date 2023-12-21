#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:4
#SBATCH -o ./SLURM.%j.out # STDOUT
#SBATCH -e ./SLURM.%j.err # STDERR
#SBATCH --cpus-per-task=10
nvidia-smi
enroot list
image_path='/home2/labhosik/llama/hee_v5.sqsh'
# parse image, container name
image_name=${image_path##*/}
container_name=${image_name%%.*}
# check old container in calculatation node and remove.
container_path="/scratch/enroot/$UID/data/$container_name"
test -d $container_path &&
rm -rf $container_path
# run container
# --mount $HOME:/mydata 는 컨테이너 외부의 $HOME 경로를
# 컨테이너 내부의 /mydata 폴더와 마운트 시킵니다.
# 적절한 경로를 마운트 하여 컨테이너 수정 없이 데이터를 옮기거나 소스코드를 수정할 수 있습니다.
enroot create -n $container_name $image_path
enroot start \
--root \
--rw \
--mount $HOME: \
$container_name \
/bin/bash -c 'cd /home2/labhosik/llama/mllama/sh && sh tllama.sh'

enroot remove --force $(enroot list)
