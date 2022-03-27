#!/bin/bash

# PI_IP=raspberrypi.local
PI_IP=navio.local
PI_USER=ubuntu
PI_PW=raspberry
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# source -> destination
#sshpass -p ${PI_PW} rsync -a $(cwd) ${PI_USER}@${PI_IP}:/home/${PI_USER}/vk_cn
rsync --exclude 'target' --exclude '.git' -avz -e ssh ${SCRIPT_DIR} ${PI_USER}@${PI_IP}:/home/${PI_USER}/
#sshpass -p '${PI_PW}' rsync --exclude 'target' --exclude '.git' -avz -e ssh ${SCRIPT_DIR} ${PI_USER}@${PI_IP}:/home/${PI_USER}/
