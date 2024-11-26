import subprocess
import json
import munch
import os
import sys

from absl import app
from absl import flags
from absl import logging as absl_logging

os.chdir(sys.path[0])

flags.DEFINE_string('test_name',   "TEST"               , 'Name of test')
# flags.DEFINE_string('config_path', './configs/Ex12DisturbOU.json', 'Path of config file')
# flags.DEFINE_string('model_name',  'NFSDE', 'name of model')
FLAGS = flags.FLAGS


def main(argv):
	del argv
	Layers = [30]
	dims = [2,5,10]
	exs = [1]


	counting = 1

	for ex in exs:
		for lay in Layers:
			for dim in dims:
				# config = munch.munchify(config)
				# config.net_config.N_epochs = epoch
				# config.net_config.l_rate = lr
				# config.net_config.l_rate_config = lrs
				# config.net_config.batch_size = bt
				# config.net_config.net_spec['nodes'] = net['n']
				# config.net_config.net_spec['layer'] = net['l']
				# config.net_config.weight_decay = wd
				# config.net_config.flevel = fl
				# config.dat_config.n_ea_traj = ds
				# json.dump(config, open(FLAGS.config_path, 'w'), indent=2)

				# cmdline  = "python3 /Users/jesse/Dropbox/NN_approximation/Program/NNapprox_tensorflow/multi_float64.py"
				# cmdline  = "python3 /home/chen.11050/NNapprox_tensorflow/multi_float64.py"
				# cmdline  = "python3 /home/chen.11050/NNapprox_tensorflow/multi_float64_arbitarystage.py"

				# cmdline  = "python3 /Users/jesse/Dropbox/NN_approximation/Program/NNapprox_tensorflow/multi_md_float64.py"
				# cmdline  = "python3 /home/chen.11050/NNapprox_tensorflow/multi_md_float64.py"
				cmdline  = "python3 /home/chen.11050/NNapprox_tensorflow/multi_md_float64_arbitarystage.py"
				cmdline += " "+str(lay)+" "+str(dim)+" "+str(ex)

				subprocess.run(cmdline, shell=True)
				counting += 1

if __name__ == '__main__':
	app.run(main)