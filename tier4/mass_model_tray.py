from argparse import ArgumentParser, Namespace
from sys import path as PYTHONPATH
from json import load, dump
from os.path import join, isdir, isfile
from os import listdir

from paramiko import SSHClient
from multiprocessing import Process, Manager

from time import time, sleep

def _exec_ssh_command(exec_info_dict, username, ip, password, command, phase):

	ssh = SSHClient()

	ssh.load_system_host_keys()
	
	ssh.connect( ip, username = username, password = password )

	stdin, stdout, stderr = ssh.exec_command( "source ~/.profile && " + command )

	#print(stdout.read()) # blocking
	
	### for debugging
	#print(stderr.read()) # blocking
	#exit_status = 0
	###

	exit_status = stdout.channel.recv_exit_status() # blocking
	
	ssh.close()

	workstation_exec_info_dict = exec_info_dict.get( username, {} )

	workstation_exec_info_dict[ phase ] = { 

		"command": command,
		"ssh_exit_status": exit_status 

	}

	exec_info_dict[ username ] = workstation_exec_info_dict

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	workstation_meta = {

		"machine70":{

			"password": "Templates12",
			"ip": "192.168.31.70"

		},

		"machine71":{

			"password": "Templates12",
			"ip": "192.168.31.71"

		},

		"machine66":{

			"password": "Templates12",
			"ip": "192.168.31.66"

		},

		"machine67":{

			"password": "Templates12",
			"ip": "192.168.31.67"

		},

		"machine68":{

			"password": "Templates12",
			"ip": "192.168.31.68"

		},

		"catalinh":{

			"password": "mirana11",
			"ip": "192.168.31.84"
		},

		"machine232":{

			"password": "Templates12",
			"ip": "192.168.31.232"
		}

	}

	_todo_trays = listdir(args.root_dir)
	
	todo_trays = []
	tray_completion_status = dict()

	for tray in _todo_trays:

		if tray == 'stash': continue
		
		tray_dir = join(args.root_dir, tray)

		if isdir(tray_dir): todo_trays.append(tray)

	available_workstations = ["catalinh", "machine70", "machine68"]
	workstation_assignments = dict()

	prep_dataset_exec_info = Manager().dict()
	model_tray_exec_info = Manager().dict()

	# TODO: check if each tray workspace meets the standard layout, i.e. prep_workspace.py or mass_prep_workspace.py did run on each tray workspace
	# TODO: have some sort of tray progression running record based on model_tray paramiko ssh command exit status, so that useless attempts will not end up discarding trays that would otherwise be tackled by the available workstations

	start = time()

	try:

		while True:

			if not todo_trays: # if there are no more trays to work on

				if not workstation_assignments.keys(): # if there are no workstation asignments

					break # job done, break out of infinite loop
				
				else:

					print('There are no more trays to work on, except for the ones currently running.')

			else:

				print(todo_trays, 'are left to do.')

			# this file will get pooled frequently for possible workstations made available in real time
			available_workstation_update_file = "/media/sml/catalinh/misc/utils/exe/tier4/samples/mass_model_tray_workstation_update.json"

			sleep(5) # pool interval in seconds

			with open(available_workstation_update_file, 'r') as fd:

				workstation_updates = load(fd)

			optional_workstations = []

			_optional_workstations = workstation_updates['workstations']

			for optional_workstation in _optional_workstations:

				if optional_workstation in available_workstations: continue

				optional_workstations.append(optional_workstation)

			available_workstations_ = available_workstations + optional_workstations

			num_workstations = len( available_workstations_ )

			for i in range(num_workstations):

				workstation = available_workstations_[ i ]

				if not workstation in list(workstation_assignments.keys()): # i.e. workstation has not been assigned to any tray workload

					if todo_trays:

						candidate = todo_trays.pop()

						print('Assigning', candidate, 'to workstation', workstation, '...')

						### compile commands

						tray_phase1_dataset = join(args.root_dir, candidate, 'workspace', 'phase1_standardized_dataset', 'subsets')
						tray_phase1_aug_config_file = join(tray_phase1_dataset, 'aug_config.json')
						
						prep_phase1_dataset_command = (
							
							"workon godapi && python /media/sml/catalinh/misc/utils/exe/tier3/prep_dataset.py -i " + 
							tray_phase1_dataset + 
							' -ac ' + 
							tray_phase1_aug_config_file
						)

						prep_phase2_dataset_command = None

						phase2_standardized_dataset_dir = join(args.root_dir, candidate, 'workspace', 'phase2_standardized_dataset')

						if isdir(phase2_standardized_dataset_dir):

							tray_phase2_aug_config_file = join(phase2_standardized_dataset_dir, 'aug_config.json')

							prep_phase2_dataset_command = ( 

								"workon godapi && python /media/sml/catalinh/misc/utils/exe/tier3/prep_dataset.py -i " + 
								phase2_standardized_dataset_dir + 
								' -ac ' + 
								tray_phase2_aug_config_file +
								'-b 16' +
								' -rec'


							)

						prep_dataset_commands = {

							"phase1": prep_phase1_dataset_command,
							"phase2": prep_phase2_dataset_command

						}

						### run commands

						workstation_info = workstation_meta.get( workstation, None )

						if workstation_info is None:

							print(workstation, 'is not a valid workstation, skipping ...')
							continue

						prep_dataset_procs = []

						print('Preping dataset for', candidate, '...')
						for phase, command in prep_dataset_commands.items():

							if command is None: continue # i.e. phase2 was not intended
							
							prep_dataset_args = (

								prep_dataset_exec_info,
								workstation, 
								workstation_info['ip'],
								workstation_info['password'],
								command,
								phase

							)

							proc = Process(

								target = _exec_ssh_command,
								args = prep_dataset_args

							)

							proc.start()
							prep_dataset_procs.append(proc)

						for proc in prep_dataset_procs: 

							proc.join()

						exec_info = prep_dataset_exec_info[ workstation ]

						phase2_exec_info = exec_info.get('phase2', None)

						failed_preping_dataset = False

						if exec_info['phase1']['ssh_exit_status'] != 0:

							failed_preping_dataset = True

						todo_phase2 = phase2_exec_info is not None

						if todo_phase2:

							if phase2_exec_info['ssh_exit_status'] != 0:
								
								failed_preping_dataset = True

						if failed_preping_dataset: 

							print('Failed preping dataset for', candidate, 'skipping ...')
							continue

						### commands finished running, dataset has been preped

						model_tray_commands = "workon godapi && python /media/sml/catalinh/misc/utils/exe/tier1/model_tray.py -i " + tray_phase1_dataset

						if todo_phase2: 

							model_tray_commands += ( 

								" && python /media/sml/catalinh/misc/utils/exe/tier1/model_tray.py -i " + 
								phase2_standardized_dataset_dir 

							)

						model_tray_args = (

							model_tray_exec_info,
							workstation, 
							workstation_info['ip'],
							workstation_info['password'],
							model_tray_commands,
							"phase1_and_phase2"

						)

						proc = Process(

							target = _exec_ssh_command,
							args = model_tray_args

						)

						proc.start()
						print('Started', candidate, 'dastaset modeling job on workstation', workstation, '...')

						# will check proc.exitcode in the else branch and react accordingly instead of waiting for proc to join here when it finishes

						workstation_assignments[ workstation ] = { 'tray_id': candidate, "process": proc  }
					
				else:

					workstation_process = workstation_assignments[ workstation ]['process']

					if workstation_process.exitcode is None: # i.e workload not completed, keep pooling for updates

						print('tray', workstation_assignments[ workstation ]['tray_id'], 'is being modeled on workstation', workstation + '@' + workstation_meta[workstation]['ip'], '...')
						sleep(3)
						continue

					else:

						workstation_assigned_tray = workstation_assignments[ workstation ]['tray_id']

						print('Finished', workstation_assigned_tray)

						del model_tray_exec_info[ workstation ] # clean any information regarding the execution that just occured on this workstation, otherwise '_for_phase2 = model_tray_exec_info.get('phase2', None)' will never yield None values for subsequent trays that might required phase2 if the current tray had work done on phase2
						del workstation_assignments[ workstation ] # make the workstation available for other trays in que for modeling

						if workstation in optional_workstations:

							optional_workstations.remove(workstation)

							with open(available_workstation_update_file, 'w') as fd:

								dump({"workstations": optional_workstations}, fd)

								todo_trays.insert(0, workstation_assigned_tray)


	except KeyboardInterrupt:

		print('Keyboard interrupt ...')

		for workstation, assignment in workstation_assignments.items():

			print('Terminating', assignment['tray_id'], 'job on workstation', workstation)

			running_job = assignment['process']
			#help(running_job)
			running_job.terminate()

			# TODO: access exec_info for both prep_dataset and model_dataset and close active ssh commands for each workstation, as of this implementation, they do not get terminated

	print('Job done, elapsed time:', (time() - start) / 60. / 60., 'hours.')
	return 0


def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory for workspace')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()