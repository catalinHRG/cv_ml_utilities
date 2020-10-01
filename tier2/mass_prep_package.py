from argparse import ArgumentParser, Namespace

from os import listdir
from os.path import join, isdir, isfile

from sys import path as PYTHONPATH

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier1')

from prep_package import run as prep_package

def run(args):

	prep_package_args = Namespace (

		p1_workspace = None,
		p2_workspace = None,
		p1_blacklist = [],
		p2_blacklist = [],
		output_dir = '/media/storagezimmer1/catalinh/biomet/models',
		tray_version = 'v1.0',
		clean = True

	)

	if not isdir(args.root_dir):

		msg = args.root_dir + ' is not a directory or does not exist.'
		raise IOError(msg)

	content = listdir(args.root_dir)

	for element in content:

		if element == 'stash': continue

		path = join(args.root_dir, element)

		if isfile(path): continue

		p1_workspace = join(path, 'workspace', 'phase1_standardized_dataset', 'subsets', 'workspace')

		if isdir(p1_workspace):

			prep_package_args.p1_workspace = p1_workspace

			p2_workspace = join(path, 'workspace', 'phase2_standardized_dataset', 'workspace')

			if isdir(p2_workspace):
 
				prep_package_args.p2_workspace = p2_workspace

			else:

				prep_package_args.p2_workspace = None # otherwise it will run with the residual value of the variable p2_workspce, from the last tray that had a phase2 to export

		_ = prep_package(prep_package_args)
			

	return 0

def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory for workspace')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()