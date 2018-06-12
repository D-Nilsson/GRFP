import os
# See "https://www.cityscapes-dataset.com/downloads/" for the zip files referenced below

# Where "gtFine_trainvaltest.zip" is unpacked
cityscapes_dir = ''

# Where "leftImg8bit_sequence_trainvaltest.zip" is unpacked. May be the same path as above.
cityscapes_video_dir = ''

# Where "https://github.com/mcordts/cityscapesScripts" is unpacked
cityscapes_scripts_root = os.path.join(cityscapes_dir, 'scripts')

example_im = os.path.join(cityscapes_dir, 'gtFine', 'train', 'aachen', 'aachen_000000_000019_gtFine_labelIds.png')
assert os.path.isfile(example_im), "The CityScapes root directory is incorrect. Could not find %s" % (example_im)

example_im = os.path.join(cityscapes_video_dir, 'leftImg8bit_sequence', 'train', 'aachen', 'aachen_000000_000000_leftImg8bit.png')
assert os.path.isfile(example_im), "The CityScapes video root directory is incorrect. Could not find %s" % (example_im)

file = os.path.join(cityscapes_scripts_root, 'evaluation', 'evalPixelLevelSemanticLabeling.py')
assert os.path.isfile(file), "Could not find the evaluation script %s" % file