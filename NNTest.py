import NNModel
from PIL import Image
import glob

for thisFile in glob.glob('./raw/train/pos/*.jpg'):
	pil_image = Image.open(thisFIle)
	thisFile.save("C:/Users/anshul/Desktop/pedestrian_detection/data/train/pos", "PIL")

for thisFile in glob.glob('./raw/train/neg/*.jpg'):
	pil_image = Image.open(thisFIle)
	thisFile.save("C:/Users/anshul/Desktop/pedestrian_detection/data/train/neg", "PIL")

for thisFile in glob.glob('./raw/test/pos/*.jpg'):
	pil_image = Image.open(thisFIle)
	thisFile.save("C:/Users/anshul/Desktop/pedestrian_detection/data/test/pos", "PIL")

for thisFile in glob.glob('./raw/test/neg/*.jpg'):
	pil_image = Image.open(thisFIle)
	thisFile.save("C:/Users/anshul/Desktop/pedestrian_detection/data/test/neg", "PIL")
