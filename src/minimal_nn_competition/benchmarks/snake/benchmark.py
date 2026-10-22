"""Load the benchmark ABC into our own benchmark with a the pass/fail criteria and score on minimum parameters"""

# Be an 8x8 grid and an agent passess if it can fill out the 8x8 grid. If it does not it gets a secondary score which allows it to be placed on the leaderboard.
# Therefore we should have {score = snake size (max 64), network_size = network size} and the ranking is score first and network size second.