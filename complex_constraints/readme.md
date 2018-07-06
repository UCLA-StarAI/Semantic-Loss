This sub-repo contains code to reproduce the complex constraints learning experiments.

The file compute_mpe.py contains everything necessary for incorporating semantic loss, and its usage can be seen in lines 65-67 of grid_net.py, or on lines 54-56 of sushi_net.py.

The arguments for running the code can be seen by running either grid_net.py or sushi_net.py with the -h flag. Both the grid and sushi experiments used MLPs with 50 hidden units per layer, with 3 hidden layers for sushi and 5 hidden layers for grids. The data for grids is found in test.data, for sushi it can be found in sushi.soc. Both have wrapper files for handling the data.

To recreate the grids experiments described in the paper, you can run:
python grid_net.py --layers 5 --units 50 --iters 50000 --data test.data --early_stopping --wmc 0.5

For the sushi experiment, run:
python sushi_net.py --layers 3 --units 25 --iters 500000 --wmc 0.25

All other configurable parameters should already be in the code (and were not finely tuned). 

The data found in test.data was generated using generate_graph_data.py. 

Finally, this project is written in python 2.7, and depends on the Python PSDD package available at https://github.com/art-ai/pypsdd (although a clone with --recursive should do this for you). The code for this package was graciously made available to us by Arthur Choi.

If you have further questions regarding our experiments or any of the code, drop me an email at tal@cs.ucla.edu.
