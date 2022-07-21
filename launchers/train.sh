#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app
launcher-xorg
python -m rl.train "${*:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
