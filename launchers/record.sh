#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# cpk run -M -f -X -L viz -A ' path/to/logdir -t ckpt'
# ckpt flag is optional and this will use the last checkpoint by default
# launching app
# launcher-xorg
python -m nlimb.record "${@:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
