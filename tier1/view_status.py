from argparse import ArgumentParser, Namespace

from os import listdir
from os.path import join, isfile, isdir, dirname, basename

from json import load, dump

from glob import glob

from time import sleep

def _get_intended_workload_on_a_per_subset_basis(model_config_file):

	per_subset_intended_configs = dict()

	with open(model_config_file) as fd:

		model_config = load(fd)

		retinanet = model_config.get('retinanet_flag', False)
		#ssd = model_config.get('ssd_flag', False)
		frcnn = model_config.get('faster_rcnn_flag', False)
		striped_frcnn = model_config.get('striped_faster_rcnn_flag', False)

	subsets_dir = dirname(model_config_file)

	_subsets = listdir(subsets_dir)
	
	subsets = []

	for subset in _subsets:

		if subset == 'workspace': continue

		path = join(subsets_dir, subset)

		if isfile(path): continue

		subsets.append(subset)

	workspace_dir = join(subsets_dir, 'workspace')

	content = listdir(workspace_dir)

	for _element in content:

		element = join(workspace_dir, _element)

		if isfile(element): continue

		if _element in subsets: 

			retinanet_workspace = join(element, 'retinanet')
			ssd_workspace = join(element, 'ssd')
			frcnn_workspace = join(element, 'frcnn')

			_configs = []

			if isdir(retinanet_workspace):

				if retinanet:

					retinanet_workspace_content = glob(join(retinanet_workspace, '*'))

					for _config in retinanet_workspace_content:

						if isdir(_config): _configs.append(_config)

			if isdir(ssd_workspace):

				if retinanet:

					ssd_workspace_content = glob(join(ssd_workspace, '*'))

					for _config in ssd_workspace_content:

						if isdir(_config): _configs.append(_config)

			if isdir(frcnn_workspace):

				frcnn_workspace_content = glob(join(frcnn_workspace, '*'))

				for _config in frcnn_workspace_content:

					if isdir(_config):

						if 'standard' in basename(_config) and frcnn:

							_configs.append(_config)

						if 'striped' in basename(_config) and striped_frcnn:

							_configs.append(_config)

			per_subset_intended_configs[ element ] = _configs
					
	return per_subset_intended_configs

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	content = listdir(args.root_dir)

	progression = dict()

	for element in content:

		if element == "stash": continue

		element_path = join(args.root_dir, element)

		if isfile(element_path): continue

		p1_subsets_dir = join( element_path, 'workspace', 'phase1_standardized_dataset', 'subsets' )
		
		_p1_subsets = listdir( p1_subsets_dir )
		
		p1_subsets = []

		for p1_subset in _p1_subsets:

			if p1_subset == 'workspace': continue

			if isdir( join( p1_subsets_dir, p1_subset ) ): p1_subsets.append( p1_subset )

		p1_workspace_dir = join( p1_subsets_dir, 'workspace' )

		p1_workspace_content = listdir( p1_workspace_dir )

		phase1_started = False

		for p1_element in p1_workspace_content:

			if isdir( join(p1_workspace_dir, p1_element) ):

				if p1_element in p1_subsets:

					phase1_started = True
					break

		progression[ element ] = { "phase1": {}, "phase2": {} }

		if phase1_started:
			
			p1_model_config_file = join(element_path, 'workspace', 'phase1_standardized_dataset', 'subsets', 'model_config.json')

			p1_intended_workloads = _get_intended_workload_on_a_per_subset_basis( p1_model_config_file )

			for subset, workload in p1_intended_workloads.items():

				progression[ element ] ['phase1'] [ subset ] = { 

					"workload": 0.0

				}

				subset_status_file = join(subset, 'status.json')

				if isfile(subset_status_file):

					with open(subset_status_file, 'r') as fd:

						_status = load(fd)

					test_precision = _status['test_precision']
					test_recall = _status['test_recall']
					train_precision = _status['train_precision']
					train_recall = _status['train_recall']

					finished_configs = 0

					for config in workload:

						if isfile(join(config, 'base', 'done_evaluating')): finished_configs += 1

					workload_completion = finished_configs / float(len(workload))

					_status['workload'] = workload_completion

					if test_precision is not None and test_recall is not None and train_precision is not None and train_recall is not None:

						if test_precision == 1.0 and test_recall == 1.0 and train_precision == 1.0 and train_recall == 1.0:

							_status['workload'] = 1.0

					progression[ element ] ['phase1'] [ subset ] = _status

			p2_workspace_dir = join(element_path, 'workspace', 'phase2_standardized_dataset')

			if isdir(p2_workspace_dir): # phase2 is intended

				phase2_started = isdir(join(p2_workspace_dir, 'workspace'))

				if phase2_started:

					p2_model_config_file = join(p2_workspace_dir, 'model_config.json')

					p2_intended_workloads = _get_intended_workload_on_a_per_subset_basis(p2_model_config_file)

					for subset, workload in p2_intended_workloads.items():

						progression[ element ] ['phase2'] [ subset ] = { 

							"workload": 0.0

						}

						subset_status_file = join(subset, 'status.json')

						if isfile(subset_status_file):

							with open(subset_status_file, 'r') as fd:

								_status = load(fd)

							test_precision = _status['test_precision']
							test_recall = _status['test_recall']
							train_precision = _status['train_precision']
							train_recall = _status['train_recall']

							if test_precision is not None and test_recall is not None and train_precision is not None and train_recall is not None:

								if test_precision == 1.0 and test_recall == 1.0 and train_precision == 1.0 and train_recall == 1.0:

									_status['workload'] = 1.0

								else:

									finished_configs = 0

									for config in workload:

										if isfile(join(config, 'base', 'done_evaluating')): finished_configs += 1

									_status['workload'] = finished_configs / float(len(workload))

							progression[ element ] ['phase2'] [ subset ] = _status

				else:

					print('Phase2 has not started yet.')

			else:

				progression[ element ] ['phase2'] = None

		else:

			print('Work on', element, 'has not started yet.')

	progress_file = join(args.root_dir, 'progress.json')

	with open(progress_file, 'w') as fd:

		dump(progression, fd, indent = 3, sort_keys = True)

def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory for tray workspaces')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()