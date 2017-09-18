# Supervied Loop Algo

## Supervied Loop Algo
Toolkits for retrieving loops from transition map.


## Usage 
* generate_loops_from_ices.py: Load icestate from h5file and create loop states and sites.
* create_markov_chain.py: Generate Type I dataset.

### Type I Data
S0 -> S -> S ...
S1 -> S -> S -> S ...

### Type II Data
S0 -> S0'
S0 -> S0''
S0 -> S0'''
...
S1 -> S1'

### Notes:
* The data are padded first.

## Credits
Thanks for fj's help.