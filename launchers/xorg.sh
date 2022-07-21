#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

set -eu

# make a fake tty
ln -s /dev/console "/dev/tty${VT}"

# launch dbus
mkdir /var/run/dbus/
dbus-daemon --system

# launching X
Xorg \
    -noreset \
    +extension GLX \
    +extension RENDER \
    -logfile "/var/log/${DISPLAY}.log" \
    -config "${XORG_CONFIG}" \
    -sharevts \
    -novtswitch \
    -verbose \
    -logverbose \
    "vt${VT}" \
    "${DISPLAY}" &
sleep 1

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
