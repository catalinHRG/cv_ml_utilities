from subprocess import Popen, PIPE
from argparse import ArgumentParser, Namespace

from sys import path as PYTHONPATH

from os.path import join, isdir

from time import time

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier3')

from prep_dataset import run as prep_dataset
	
model_tray_utility_dir = '/media/sml/catalinh/misc/utils/exe/tier1'

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory'
		raise IOError(msg)

	if not args.tray_ids:

		msg = 'There has to be at least one tray id to be found in args.root_dir in order to process'
		raise ValueError(msg)

	try:

		log_file = join(args.root_dir, args.log_file_name)

		while True:

			proc = None
			
			for tray_id in args.tray_ids:

				tray_dir = join(args.root_dir, tray_id)

				if not isdir(tray_dir):

					msg = tray_id + ' cannot be found in ' + args.root_dir + ' skipping ...'
					
					print( msg )

					continue

				p1_elapsed_time, p2_elapsed_time = 0, 0

				tray_phase1_workspace_dir = join(tray_dir, 'workspace', 'phase1_standardized_dataset', 'subsets')

				if isdir(tray_phase1_workspace_dir):

					print('preping phase1 dataset for', tray_id, '...')

					prep_dataset_args = Namespace (

						root_dir = tray_phase1_workspace_dir,
						aug_config = join(tray_phase1_workspace_dir, 'aug_config.json'),
						base = 64,
						min_obj_size = 45,
						max_obj_size = 50,
						include_rectagular = False

					)

					_ = prep_dataset(prep_dataset_args)

					print('modeling phase1 for', tray_id, '...')

					# the reason why model_tray has not been imported and ran as a python module is because if the code within it leads to some abusive behavior, by being terminated, we bypass this process invoking it as a bash command of being terminated aswell, thus alowing it to continuously invoke it
					#proc = Popen( [ 'source ~/.profile && workon godapi && python', join( model_tray_utility_dir, 'model_tray.py' ), '-i', tray_phase1_workspace_dir ], bufsize = 2048, stdin = PIPE )
					
					command = ['python', join( model_tray_utility_dir, 'model_tray.py' ), '-i', tray_phase1_workspace_dir ]

					if args.skip_config_generation:

						command += '-s'

					p1_start_time = time()

					proc = Popen( command )
					
					proc.wait()

					p1_elapsed_time = time() - p1_start_time

					with open(log_file, 'a') as fd:

						fd.write( tray_id + ' phase1 elapsed time: ' + str( p1_elapsed_time / 3600 ) + ' hours\n' )

				else:

					msg = tray_id + ' phase1 workspace dir does not exist: ' + tray_phase1_workspace_dir + ' skipping ...'
					
					print( msg )

					continue

				tray_phase2_workspace_dir = join(tray_dir, 'workspace', 'phase2_standardized_dataset')

				if isdir(tray_phase2_workspace_dir):

					print('preping phase2 dataset for', tray_id, '...')

					prep_dataset_args = Namespace (

						root_dir = tray_phase2_workspace_dir,
						aug_config = join(tray_phase2_workspace_dir, 'aug_config.json'),
						base = 16,
						min_obj_size = 50,
						max_obj_size = 60,
						include_rectagular = True

					)

					_ = prep_dataset(prep_dataset_args)

					print('modeling phase2 for', tray_id, '...')

					# the reason why model_tray has not been imported and ran as a python module is because if the code within it leads to some abusive behavior, by being terminated, we bypass this process invoking it as a bash command of being terminated aswell, thus alowing it to continuously invoke it
					#proc = Popen( [ 'source ~/.profile && workon godapi && python', join( model_tray_utility_dir, 'model_tray.py' ), '-i', tray_phase2_workspace_dir], bufsize = 2048, stdin = PIPE )
					
					command = ['python', join( model_tray_utility_dir, 'model_tray.py' ), '-i', tray_phase2_workspace_dir ]

					if args.skip_config_generation:

						command += '-s'

					p2_start_time = time()

					proc = Popen( command )

					proc.wait()

					p2_elapsed_time = time() - p2_start_time

					with open(log_file, 'a') as fd:

						fd.write( tray_id + ' phase2 elapsed time: ' + str( p2_elapsed_time / 3600 ) + ' hours\n' )

				else:

					msg = tray_id + ' phase2 workspace dir does not exist: ' + tray_phase2_workspace_dir + ' skipping ...'
					
					print( msg )

					continue

	except KeyboardInterrupt:

		print('keyboard interupt, aborting ...')

		if proc: 

			proc.kill()

		exit()

	return 0

def main():

    parser = ArgumentParser( description = 'This utility will invoke model_tray.py in an infinite loop for several trays given as input, reason being, bypas any unforseen contingencies like memory access violations/segmentation faults arisen from the apis invoked throught the involved modules' )
    
    parser.add_argument( '--root_dir', '-i', required = True, help = 'Root directory path for standardized dataset layout.' )
    parser.add_argument( '--tray_ids', '-t', nargs = '*', default = [], dest = 'tray_ids', help = 'list of tray ids to be modeled')
    parser.add_argument( '--skip_config_generation', '-s', action = 'store_true', help = 'skips the pipeline config files generation. this is mainly to be used in case there was not enough memory for the baseline configurations to run.')
    parser.add_argument( '--log_file_name', '-l', required = False, default = 'log.txt', help = 'file name used in order to log tray timers')

    args = parser.parse_args()

    status = run(args)

if __name__ == '__main__': main()