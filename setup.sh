#!/bin/bash

# NOTE: this setup script will be executed right before the launcher file inside the container,
#       use it to configure your environment.

set -eu

# constants
export VT=1
export XORG_CONFIG=/assets/xorg.nvidia.conf
export BUS_ID=`python get_bus_id.py`
git config --global --add safe.directory /code/src/nlimb-devel

if [ -z ${BUS_ID+x} ]; then
    echo "Environment variable BUS_ID is not set. Use 'nvidia-xconfig --query-gpu-info' to find the PCI Bus ID of the GPU you want to use."
else
    echo "BUS_ID is set to '$BUS_ID'"
    # replace placeholder BUS_ID
    sed -i "s/NVIDIA_PCI_BUS_ID/${BUS_ID}/g" "${XORG_CONFIG}"
fi

# set up wandb authentication
if test -f "/root/wandb_info/.netrc"; then
    cp "/root/wandb_info/.netrc" "/root/.netrc"
fi

set +eu
    
