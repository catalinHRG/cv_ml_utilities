from argparse import ArgumentParser, Namespace

from os import listdir
from os.path import join, isfile, isdir

from json import load, dump

p1_model_config = {

   "blacklist": [],
   "faster_rcnn_flag": True,
   "max_steps_per_attempt": 1000,
   "num_interpolated_convs_between_encoding_and_predictor": 2,
   "retinanet_flag": False,
   "striped_faster_rcnn_flag": True,
   "target": [],
   "unmatched_threshold_values": [
      0.5,
      0.45,
      0.4,
      0.35,
      0.3
   ]
}

p2_model_config = {

   "blacklist": [],
   "faster_rcnn_flag": True,
   "max_steps_per_attempt": 1000,
   "num_interpolated_convs_between_encoding_and_predictor": 2,
   "retinanet_flag": False,
   "striped_faster_rcnn_flag": True,
   "target": [],
   "unmatched_threshold_values": [
      0.5,
      0.45,
      0.4,
      0.35,
      0.3
   ]
}

def _update_config(model_config_file, model_config):

	if isfile(model_config_file):

		print('updating', model_config_file)

		update = False

		with open(model_config_file, 'r') as fd:

			current_config = load(fd)

			if current_config: update = True

		# model_config_file has been closed

		if update:

			with open(model_config_file, 'w') as fd:

				current_config['retinanet_flag'] = True
				current_config['faster_rcnn_flag'] = True
				current_config['max_steps_per_attempt'] = 1500

				dump(current_config, fd, indent = 3, sort_keys = True)

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	content = listdir(args.root_dir)

	for element in content:

		if element == "stash": continue

		element_path = join(args.root_dir, element)

		if isfile(element_path): continue

		p1_model_config_file = join(element_path, 'workspace', 'phase1_standardized_dataset', 'subsets', 'model_config.json')

		_update_config(p1_model_config_file, p1_model_config)
		
		p2_model_config_file = join(element_path, 'workspace', 'phase2_standardized_dataset', 'model_config.json')

		_update_config(p2_model_config_file, p2_model_config)


def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory for tray workspaces')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()