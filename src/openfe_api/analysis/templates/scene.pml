# $title
#
# Open with:  pymol $structure
#
# The frame whose ligand sits closest to the median pose of the run, so it represents the
# run rather than wherever the simulation happened to stop.

load $structure, scene

hide everything
bg_color white
set ray_opaque_background, 0

show cartoon, scene and polymer
color grey80, scene and polymer
set cartoon_transparency, 0.45

show sticks, scene and resn UNK
color marine, scene and resn UNK and element C
util.cnc("scene and resn UNK")

select pocket, byres (scene and polymer within $radius of (scene and resn UNK))
show sticks, pocket and sidechain
color wheat, pocket and element C
set stick_radius, 0.12, pocket

orient scene and resn UNK
zoom scene and resn UNK, 6
deselect
