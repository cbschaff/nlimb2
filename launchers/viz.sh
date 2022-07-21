#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# cpk run -M -f -X -L viz -A ' path/to/logdir -t ckpt -n num_designs --mode'
# ckpt flag is optional and this will use the last checkpoint by default
python -m nlimb.viz "${@:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
