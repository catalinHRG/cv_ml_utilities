from argparse import ArgumentParser

from os import listdir, mkdir, system

from os.path import join, isfile, isdir, dirname, basename

from sys import path as PYTHONPATH

from copy import deepcopy

from json import load, dump

from subprocess import Popen, PIPE

from time import sleep

PYTHONPATH.insert(0, "/media/sml/catalinh/misc/godapi/research/object_detection")

from model_main import run as model_dataset

from utils.config_util import get_configs_from_pipeline_file as create_config
from utils.config_util import create_pipeline_proto_from_configs as create_pipeline
from utils.config_util import save_pipeline_config as save_pipeline

def put_quotes(string):

	return '"' + string + '"'


def _fetch_tray_info(workspace, blacklist):

	tray_info = dict()

	workspace_content = listdir( workspace )

	for element in workspace_content:

		if element in blacklist: continue

		path_ = join( workspace, element )

		if not isdir( path_ ): continue

		status_file = join(path_, 'status.json')

		solution_pointer = {}

		if isfile(status_file): 

			with open(status_file, 'r') as fd:

				solution_pointer = load(fd)

		tray_info[ element ] = {

			"solution_pointer": solution_pointer

		}

	return tray_info

def _fetch_subset_stats(subset, workspace):

	box_stats_dir = join(dirname(workspace), subset, 'info')

	if not isdir(box_stats_dir):

		msg = 'Cant\'t find box stats directory relative to p1 workspace'
		raise IOError(msg)

	box_stats_file = join(box_stats_dir, 'box_stats.json')
	stats_file = join(box_stats_dir, 'stats.json')

	if not isfile(box_stats_file):

		msg = box_stats_file + ' does not exist or is not a file'
		raise IOError(msg)

	with open(box_stats_file, 'r') as fd:

		box_stats = load(fd)

	if not isfile(stats_file):

		msg = stats_file + ' does not exist or is not a file'
		raise IOError(msg)

	with open(stats_file, 'r') as fd:

		stats = load(fd)

	return stats, box_stats


def _export(

		subset, 
		workbench, 
		checkpoint, 
		meta_arch, 
		max_number_of_boxes_per_image, 
		max_number_of_boxes_per_class_per_image, 
		clean

	):

	print('Subset', subset, '...')

	retinanet_flag, frcnn_flag = False, False 

	retinanet_flag = True if meta_arch == 'retinanet' else False
	frcnn_flag = True if meta_arch == 'frcnn' else False

	_export_config = create_config(join(workbench, 'eval.config'))
	
	export_config = deepcopy(_export_config)

	if retinanet_flag:

		if max_number_of_boxes_per_image == 1:

			export_config['model'].ssd.post_processing.batch_non_max_suppression.iou_threshold = 0.05
		
		export_config['model'].ssd.post_processing.batch_non_max_suppression.max_total_detections = max_number_of_boxes_per_image
		export_config['model'].ssd.post_processing.batch_non_max_suppression.max_detections_per_class = max_number_of_boxes_per_class_per_image

	if frcnn_flag:

		if max_number_of_boxes_per_image == 1:

			export_config['model'].faster_rcnn.second_stage_post_processing.batch_non_max_suppression.iou_threshold = 0.05
		
		export_config['model'].faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = max_number_of_boxes_per_image
		export_config['model'].faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = max_number_of_boxes_per_class_per_image

	pipeline = create_pipeline( export_config )

	save_pipeline( pipeline, workbench, 'eval.config' )

	export_folder = join( workbench, 'export' )

	if clean:
		
		if isdir(export_folder): # export util will not work if there is an export folder already
			
			p = Popen( ['rm', '-r', export_folder], bufsize = 2048, stdin = PIPE)
			p.wait()

	p = Popen( ['godapiExport', workbench, str(checkpoint)], bufsize = 2048, stdin = PIPE)
	p.wait()

	return _export_config


def run(args):

	'''

		P1:

			for every datasubset found in p1 workspace:

				if no solution and it's not blacklisted: 
				
					mark as incomplete (will abort after this sweep and inform user about incomplete subsets so that they can be investigated)
				
				else: 
				
					mark as complete

			abort if tray is incomplete

			for every datasubset found in p1 workspace:

				modify eval.config @ nms params to suit the datasubset
				godapiExport invoked via shell using the params found in each status.json (pointers to solution i.e. checkpoint root_dir and checkpoint name)
				
				copy over to according subset dir in package_generator compatible layout within the 'partitioned' section, according to the specified tray version associated with this partition scheme
		
		P2:

			todo		

	'''

	include_phase2 = False

	if not isdir(args.p1_workspace):

		msg = args.p1_workspace + ' does not exist or is not a directory.'
		raise IOError(msg)

	if args.p2_workspace is not None: # this can be optional, some tray dont need p2

		if not isdir(args.p2_workspace):

			msg = args.p2_workspace + ' does not exist or is not a directory.'
			raise IOError(msg)

		include_phase2 = True

	if not isdir(args.output_dir):

		msg = args.output_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	tray_id = basename( dirname(dirname(dirname(dirname(args.p1_workspace)))) )

	tray_dir = join(args.output_dir, tray_id)
	if not isdir(tray_dir): mkdir(tray_dir)

	phase1_dir = join(tray_dir, 'phase1')
	if not isdir(phase1_dir): mkdir(phase1_dir)

	partitioning_dir = join(phase1_dir, 'partitioned')
	if not isdir(partitioning_dir): mkdir(partitioning_dir)

	partition_version_dir = join(partitioning_dir, args.tray_version)
	if not isdir(partition_version_dir): mkdir(partition_version_dir)

	groups_lmap_file_path = join(args.p1_workspace, 'groups_lmap.pbtxt')

	#command = 'cp ' + put_quotes(groups_lmap_file_path) + ' ' + put_quotes(partition_version_dir)
	#system(command)

	proc = Popen( [ 'cp', groups_lmap_file_path, partition_version_dir ], bufsize = 2048, stdin = PIPE)
	proc.wait()

	print('Sweeping for phase1 subsets status ...')

	p1_tray_info = _fetch_tray_info(args.p1_workspace, args.p1_blacklist)

	incomplete_subsets = []
	for subset, info in p1_tray_info.items():

		if not info: 

			incomplete_subsets.append(subset)

		else:

			workbench = info['solution_pointer'].get('workbench', None)
			
			if workbench is None:

				incomplete_subsets.append(subset)

	if incomplete_subsets:

		msg = 'The following subsets are not finished:'
		for subset in incomplete_subsets:
			msg = msg + ' ' + subset + ' '

		raise ValueError(msg)

	if include_phase2:

		print('Sweeping for phase2 subsets status ...')

		p2_tray_info = _fetch_tray_info(args.p2_workspace, args.p2_blacklist)

		incomplete_subsets = []
		for subset, info in p2_tray_info.items():

			if not info: 

				incomplete_subsets.append(subset)

			else:

				workbench = info['solution_pointer'].get('workbench', None)

				if workbench is None:

					incomplete_subsets.append(subset)


		if incomplete_subsets:

			msg = 'The following subsets are not finished:'
			for subset in incomplete_subsets:
				msg = msg + ' ' + subset + ' '

			raise ValueError(msg)

	print('Exporting phase1 subsets ...')

	for subset, info in p1_tray_info.items():

		solution_pointer = info['solution_pointer']
		workbench = solution_pointer['workbench']

		meta_arch = basename(dirname(dirname(workbench)))

		stats, box_stats = _fetch_subset_stats(subset, args.p1_workspace)

		original_image_height = stats['original_image_height']
		original_image_width = stats['original_image_width']

		max_number_of_boxes_per_image = box_stats['max_box_count_per_image']
		max_number_of_boxes_per_class_per_image = box_stats['max_box_count_per_image_per_class'] 

		solution_pointer = info['solution_pointer']
		workbench = solution_pointer['workbench']

		export_config = _export( # helper

			subset, 
			workbench, 
			solution_pointer['checkpoint'],
			meta_arch, 
			max_number_of_boxes_per_image, 
			max_number_of_boxes_per_class_per_image,
			args.clean

		)

		subset_dir = join(partition_version_dir, subset)

		if not isdir(subset_dir):

			mkdir(subset_dir)

			latest_subset_version = join(subset_dir, 'v1.0')
			mkdir(latest_subset_version)

		else:

			current_versions = listdir(subset_dir) 
			current_versions.sort()

			latest_subset_version = current_versions[-1]

			latest_subset_version = "v" + str(int(latest_subset_version[1]) + 1) + '.0'
			latest_subset_version = join(subset_dir, latest_subset_version)

			mkdir(latest_subset_version)

		export_folder = join( workbench, 'export' )

		proc = Popen( ['cp', '-r', export_folder, latest_subset_version], bufsize = 2048, stdin = PIPE)
		proc.wait()

		if proc.returncode == 0: # i.e. success
		    
			final_export_folder = join(latest_subset_version, 'export')

			lmap_file_path = export_config['train_input_config'].label_map_path

			#command = 'cp ' + put_quotes(lmap_file_path) + ' ' + put_quotes(final_export_folder)
			#system(command)

			proc = Popen( [ 'cp', lmap_file_path, final_export_folder ], bufsize = 2048, stdin = PIPE)
			proc.wait()

			subset_runtime_info = {

				"detector_input_width": original_image_width,
				"detector_input_height": original_image_height,
				"detector_threshold": 0.3

			}

			with open(join(final_export_folder, 'info.json'), 'w') as fd:

				dump(subset_runtime_info, fd)

		# revert to original eval.config inteded for evaluation

		pipeline = create_pipeline( export_config )

		save_pipeline( pipeline, workbench, 'eval.config' )

	if include_phase2:

		print('Exporting phase2 subsets ...')

		phase2_dir = join(tray_dir, 'phase2')
		if not isdir(phase2_dir): mkdir(phase2_dir)

		for subset, info in p2_tray_info.items():

			solution_pointer = info['solution_pointer']
			workbench = solution_pointer['workbench']

			meta_arch = basename(dirname(dirname(workbench)))

			stats, box_stats = _fetch_subset_stats(subset, args.p2_workspace)

			original_image_height = stats['original_image_height']
			original_image_width = stats['original_image_width']

			max_number_of_boxes_per_image = box_stats['max_box_count_per_image']
			max_number_of_boxes_per_class_per_image = box_stats['max_box_count_per_image_per_class'] 

			export_config = _export( # helper

				subset, 
				workbench, 
				solution_pointer['checkpoint'],
				meta_arch, 
				1, 
				1,
				args.clean

			)

			subset_dir = join(phase2_dir, subset)

			if not isdir(subset_dir): mkdir(subset_dir)

			_subset_dir = join(subset_dir, 'det')

			if not isdir(_subset_dir):

				if not isdir(_subset_dir): mkdir(_subset_dir)

				latest_subset_version = join(_subset_dir, 'v1.0')
				mkdir(latest_subset_version)

			else:

				current_versions = listdir(_subset_dir) 
				current_versions.sort()

				latest_subset_version = current_versions[-1]

				latest_subset_version = "v" + str(int(latest_subset_version[1]) + 1) + '.0'
				latest_subset_version = join(_subset_dir, latest_subset_version)

				mkdir(latest_subset_version)

			export_folder = join( workbench, 'export' )

			proc = Popen( ['cp', '-r', export_folder, latest_subset_version], bufsize = 2048, stdin = PIPE)
			proc.wait()

			if proc.returncode == 0: # i.e. success
			    
				final_export_folder = join(latest_subset_version, 'export')

				lmap_file_path = export_config['train_input_config'].label_map_path

				#command = 'cp ' + put_quotes(lmap_file_path) + ' ' + put_quotes(final_export_folder)
				#system(command)

				proc = Popen( [ 'cp', lmap_file_path, final_export_folder ], bufsize = 2048, stdin = PIPE)
				proc.wait()

				subset_runtime_info = {

					"detector_input_width": original_image_width,
					"detector_input_height": original_image_height,
					"detector_threshold": 0.1

				}

				with open(join(final_export_folder, 'info.json'), 'w') as fd:

					dump(subset_runtime_info, fd)

			# revert to original eval.config inteded for evaluation

			pipeline = create_pipeline( export_config )

			save_pipeline( pipeline, workbench, 'eval.config' )

	return 0

def main():

	parser = ArgumentParser( description = 'Utility used for generating the layout required by the package generation utility, given workspaces for both p1 and p2, containing trained godapi models' )
	
	parser.add_argument( '--p1_workspace', '-p1w', required = True, help = 'Root directory path for a phase1 workspace layout coresponding to model_tray.py' )
	parser.add_argument( '--p2_workspace', '-p2w', required = False, help = 'Root directory path for a phase2 workspace layout coresponding to model_tray.py' )
	parser.add_argument( '--p1_blacklist', '-p1b', nargs = '*', default = [], dest = 'p1_blacklist', help = 'Enables user to blacklist certain subsets when generating the required layout for package_generator, space separated values')
	parser.add_argument( '--p2_blacklist', '-p2b', nargs = '*', default = [], dest = 'p2_blacklist', help = 'Enables user to blacklist certain subsets when generating the required layout for package_generator, space separated values')
	parser.add_argument( '--output_dir', '-o', required = False, default = '/media/storagezimmer1/catalinh/biomet/models', help = 'Output directory path where the layout expected by package generator is to be created')
	parser.add_argument( '--tray_version', '-tv', required = False, type = str, default = 'v1.0', help = 'Version to be used for the various partitioning schemes found at a tray level')
	parser.add_argument( '--clean', '-c', action = 'store_true', help = 'clean workspaces of any previous exports')

	args = parser.parse_args()

	_ = run(args)

if __name__ == '__main__': main()