#!/bin/bash

# PI_IP=raspberrypi.local
PI_IP=navio.local
PI_USER=ubuntu
PI_PW=raspberry
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# source -> destination
rsync --exclude 'target' --exclude 'vkcv/out' --exclude '.git' -avz -e ssh ${SCRIPT_DIR} ${PI_USER}@${PI_IP}:/home/${PI_USER}/

