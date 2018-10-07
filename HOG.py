import numpy as np
import matplotlib.pyplot as plt
def pngToGray(filename):
	rgb_image = plt.imread(filename)
	gray_image = 0.299*rgb_image[:,:,0] + 0.587 *rgb_image[:,:,1] + 0.114 *rgb_image[:,:,2]
	return gray_image

#assumes images are 96x160 as in test and train 64x128 folders
#assumes images are inputed as grayscales numpy array
#TODO look at np.array_split
def splitIntoCells(image): 
	#trim to 66x130 (edge pixels are used for gradient detection)
	image = image[14:80,14:144]
	#Corner cells should be 9x9, side cells should be 8x9, top/bottom cells should be 9x8, middle cells should be 8x8
def calculateGradientVals(cell):
	pass
def caclulateHistogram(cell):
	pass
def normalize_blocks(vec):
	pass

