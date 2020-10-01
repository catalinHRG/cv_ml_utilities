#!/bin/bash

#common=/media/sml2/catalinh/biomet/models/det/phase2/2CL04011007/export
#src_dir=/media/sml2/catalinh/biomet/data/phase2/orig/2CL04011007/Crops/tool_category_2/resized/512x400/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase2/orig/2CL04011007/Crops/tool_category_2/lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase2/2CL03011088/tool_category_1/export
#src_dir=/media/sml2/catalinh/biomet/data/phase2/orig/bundles/1/tool_category_1/resized/640x176/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase2/orig/bundles/1/tool_category_1/2CL03011088_lmap.pbtxt

#src_dir=/media/sml2/catalinh/biomet/data/phase2/orig/2CL03011088/Crops/tool_category_1/resized/640x176/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase2/orig/2CL03011088/Crops/tool_category_1/lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase2/2CL02011002/tool_category_2/export
#src_dir=/media/sml2/catalinh/biomet/data/phase2/orig/2CL02011002/Crops/tool_category_2/resized/224x640/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase2/orig/2CL02011002/Crops/tool_category_2/lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase2/2CL01011517l/tool_category_1/export
##src_dir=/media/sml2/catalinh/biomet/data/phase2/orig/bundles/2/resized/512x256/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase2/orig/bundles/2/resized/2CL01011517l_tool_category_1_lmap.pbtxt

#common=/media/machine70/WORK/catalinh/workspace/biomet/models/v2.6/export
#common=/media/sml/catalinh/biomet/models/deploy/tray/v2.7/export
#src_dir=/media/machine70/WORK/catalinh/workspace/biomet/data/bundled/resized/320x160/$3


#common=/media/sml2/catalinh/biomet/models/det/tray/frcnn/export
#common=/media/machine70/WORK/catalinh/workspace/biomet/models/v2.7.2/export
#src_dir=/media/machine70/WORK/catalinh/workspace/biomet/data/bundled/$3/resized/320x180

#src_dir=/media/sml2/catalinh/biomet/data/phase1/group1/root/2CL03011005/resized/512x256/train
#src_dir=/media/machine70/WORK/catalinh/workspace/biomet/data/holdout
#lmap_file=/media/machine70/WORK/catalinh/workspace/biomet/data/lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/1/root/2CL03011003//export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/group1/root/2CL03011003//resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/group1/root/2CL03011003//groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/6/1CL02015077/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/group6/1CL02015077/resized/320x160/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/group6/1CL02015077/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/6/1CL02015227_SizeH/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/group6/1CL02015227_Size$3/resized/320x160/$4
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/group6/1CL02015227_SizeH/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/1/root/bundle2/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group1/root/bundles/1/all/$3
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group1/root/bundles/1/2CL03011005/resized/1024x512/$3
#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/1/root/bundle3/workspace/atrous1/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group1/root/bundles/2/relevant/$3
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group1/root/bundles/2/relevant/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/groups/group1/root/2CL03011003/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/8/export
#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/8/1CL02012203u/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/group8/1CL02012203u/resized/320x160/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/group8/1CL02012203u/groups_lmap.pbtxt
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups8/1CL02012203$3/resized/320x160/$4
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/groups8/1CL02012203u/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/8/1CL02012203l/v3.0/export
#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/8/1CL02012203l/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/group8/1CL02012203$3/resized/640x320/$4
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group8/1CL02012203$3/resized/640x320/$4
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/group8/1CL02012203l/groups_lmap.pbtxt
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/groups/group8/1CL02012203l/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/1CL02015116/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02015116/resized/1024x1024/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02015116/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/groups/11/3CL04015016l/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/groups/group10/3CL04015016l/resized/640x320/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/groups/group10/3CL04015016l/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/2CL01012023/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01012023/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01012023/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/3CL04015182/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/3CL04015182/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/3CL04015182/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/1CL02012223/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012223/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012223/groups_lmap.pbtxt


#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/2CL01011029/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011029/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011029/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/2CL01011225/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011225/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011225/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/1CL02012375u/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012375u/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012375u/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/2CL01012023/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01012023/resized/_1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01012023/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/workbench/1CL02012423/frcnn/0.75depth/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012423/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/1CL02012423/groups_lmap.pbtxt

#common=/media/sml2/catalinh/biomet/models/det/phase1/leftovers/workbench/2CL01011327u/frcnn/1.0depth/export
#src_dir=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011327u/resized/1024x512/$3
#lmap_file=/media/sml2/catalinh/biomet/data/phase1/leftovers/2CL01011327u/groups_lmap.pbtxt

python inference.py -lm "$3" \
		-o "$2" \
		-i "$4" \
		-m "$1/frozen_inference_graph.pb" \
		-iou $5 -obj $6



