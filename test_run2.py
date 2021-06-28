

from raindrop.dropgenerator import generateDrops

cfg = {
	'maxR': 50,
	'minR': 30,
	'maxDrops': 30,
	'minDrops': 30,
	'edge_darkratio': 0.3,
	'return_label': True,
	'label_thres': 128
}

# it will return image in pillow format
# if using cfg["return_label"] = False
output_image = generateDrops("Picture1.png", cfg)

# if using cfg["return_label"] = True
output_image, output_label = generateDrops("Picture1.png", cfg)