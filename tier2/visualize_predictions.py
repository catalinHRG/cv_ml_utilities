from sys import path as PYTHONPATH
from argparse import ArgumentParser, Namespace
from json import load
from os.path import isdir, isfile, join, dirname, basename
from subprocess import Popen, PIPE
from os import mkdir

PYTHONPATH.insert(0, "/media/sml/catalinh/misc/godapi/research/object_detection")

from utils.config_util import get_configs_from_pipeline_file as create_config


def _fetch_args_from_pipeline(pipeline_config_file):

	config = create_config(pipeline_config_file)

	lmap_file = config['train_input_config'].label_map_path
	
	if not isfile(lmap_file):

		msg = lmap_file + ' does not exist or is not a file'
		raise IOError(msg)

	train_tfr_regex_path = str(config['train_input_config'].tf_record_input_reader.input_path[0]) # the root dir for it is the root dir for train and eval directories also, where the images are to be found

	train_input_dir = join(dirname(train_tfr_regex_path), 'train')
	eval_input_dir = join(dirname(train_tfr_regex_path), 'eval')

	if not isdir(train_input_dir):

		msg = train_input_dir + ' does not exist or is not a directory'
		raise IOError(msg)

	if not isdir(eval_input_dir):

		msg = eval_input_dir + ' does not exist or is not a directory'
		raise IOError(msg)

	return lmap_file, train_input_dir, eval_input_dir

def _parse_info(info, output_dir):

	for subset_workspace, status in info.items():

		subset = basename(subset_workspace)

		checkpoint = status.get('checkpoint', None)
		test_precision = status.get('test_precision', None)
		test_recall = status.get('test_recall', None)
		train_precision = status.get('train_precision', None)
		train_recall = status.get('train_recall', None)
		workbench = status.get('workbench', None)

		if checkpoint is not None and test_precision is not None and test_recall is not None and train_precision is not None and train_recall is not None:

			if test_precision < 1.0 or test_recall < 1.0 or train_precision < 1.0 or train_recall < 1.0:

				subset_output_dir = join(output_dir, subset)
		
				if not isdir(subset_output_dir): mkdir(subset_output_dir)

				export_dir = join(workbench, 'export')

				if isdir(export_dir):

					proc = Popen( ['rm', '-r', export_dir] )
					proc.wait()

				proc = Popen( ['godapiExport', workbench, str(checkpoint)] )
				proc.wait()

				model_file = join(export_dir, 'frozen_inference_graph.pb')

				if not isfile(model_file):

					msg = model_file + ' does not exist or is not a file'
					raise IOError(msg)

				pipeline_config_file = join(workbench, 'export', 'pipeline.config')

				if not isfile(pipeline_config_file): # i.e. was deleted or export did not finish properly

					msg = pipeline_config_file + ' does not exist or is not a file'
					raise IOError(msg)
							
				lmap_file, train_input_dir, eval_input_dir = _fetch_args_from_pipeline( pipeline_config_file )

				if test_precision < 1.0:
					
					obj_threshold = 0.1
					
					_output_dir = join(subset_output_dir, 'eval_obj_th_' + str(obj_threshold))

					if not isdir(_output_dir): mkdir(_output_dir)

					# run inference for each split and the according obj thresholds found in todos

					proc = Popen( [ 'python', '/media/sml/catalinh/misc/utils/exe/tier1/inference.py', '-m', model_file, '-o', _output_dir, '-lm', lmap_file, '-i', eval_input_dir, '-iou', '0.1', '-obj', str(obj_threshold) ] )
					proc.wait()

				if test_recall < 1.0:
					
					obj_threshold = 0.9
					
					_output_dir = join(subset_output_dir, 'eval_obj_th_' + str(obj_threshold))

					if not isdir(_output_dir): mkdir(_output_dir)

					# run inference for each split and the according obj thresholds found in todos
					proc = Popen( [ 'python', '/media/sml/catalinh/misc/utils/exe/tier1/inference.py', '-m', model_file, '-o', _output_dir, '-lm', lmap_file, '-i', eval_input_dir, '-iou', '0.1', '-obj', str(obj_threshold) ] )
					proc.wait()

				if train_precision < 1.0:
					
					obj_threshold = 0.1
					
					_output_dir = join(subset_output_dir, 'eval_obj_th_' + str(obj_threshold))

					if not isdir(_output_dir): mkdir(_output_dir)

					# run inference for each split and the according obj thresholds found in todos
					proc = Popen( [ 'python', '/media/sml/catalinh/misc/utils/exe/tier1/inference.py', '-m', model_file, '-o', _output_dir, '-lm', lmap_file, '-i', train_input_dir, '-iou', '0.1', '-obj', str(obj_threshold) ] )
					proc.wait()

				if train_precision < 1.0:
					
					obj_threshold = 0.9
					
					_output_dir = join(subset_output_dir, 'eval_obj_th_' + str(obj_threshold))

					if not isdir(_output_dir): mkdir(_output_dir)

					# run inference for each split and the according obj thresholds found in todos
					proc = Popen( [ 'python', '/media/sml/catalinh/misc/utils/exe/tier1/inference.py', '-m', model_file, '-o', _output_dir, '-lm', lmap_file, '-i', train_input_dir, '-iou', '0.1', '-obj', str(obj_threshold) ] )
					proc.wait()


def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory'
		raise IOError(msg)

	if args.output_dir is not None:

		output_dir = args.output_dir

		if not isdir(args.output_dir):

			msg = args.output_dir + ' does not exist or is not a directory'
			raise IOError(msg)

	else:

		output_dir = join(args.root_dir, 'stash')

		if not isdir(output_dir): mkdir(output_dir)

		output_dir = join(output_dir, 'prediction')

		if not isdir(output_dir): mkdir(output_dir)

	progress_file = join(args.root_dir, 'progress.json')
	
	if not isfile(progress_file):

		msg = progress_file + ' does not exist or is not a file'
		raise IOError(msg)

	with open(progress_file, 'r') as fd:

		progress = load(fd)

	for tray_id, info in progress.items():

		if args.whitelist:

			if tray_id not in args.whitelist: continue

		_output_dir = join(output_dir, tray_id)
		
		if not isdir(_output_dir): mkdir(_output_dir)

		p1_output_dir = join(_output_dir, 'phase1')

		if not isdir(p1_output_dir): mkdir(p1_output_dir)

		phase1_info = info.get('phase1', {})

		_parse_info( phase1_info, p1_output_dir )

		phase2_info = info.get('phase2', {})

		if phase2_info is not None:

			p2_output_dir = join(_output_dir, 'phase2')

			if not isdir(p2_output_dir): mkdir(p2_output_dir)

			_parse_info( phase2_info, p2_output_dir)

	return 0

def main():

	parser = ArgumentParser(description = 'Visualize predictions for the underperforming subsets found in the progress.json file yielded by the visualize_status utility')
	
	parser.add_argument('--root_dir', '-i', required = True, help = 'Root directory path where tray workspaces are to be found along with a progression json file which contains potential solution pointers')
	parser.add_argument('--output_dir', '-o', required = False, help = 'Root directory path used to dump predictions for visualization')
	parser.add_argument('--whitelist', '-w', nargs = '*', default = [], dest = 'whitelist', help = 'list of tray id subset to run on')

	args = parser.parse_args()

	_ = run(args)

if __name__ == '__main__': main()