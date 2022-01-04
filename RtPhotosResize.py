import glob, os, time
from PIL import Image

serverEpiFolder = "C:\\Users\\lucia\\Desktop\\EpidemicModel\\"
RtFolder = serverEpiFolder+'WorkingDataset\ItalyInteractiveMap_Active\it-js-map\RtFolder\\'
os.chdir(RtFolder)

for fileIn in glob.glob("*.png"):
    image = Image.open(fileIn)
    image = image.resize((500, 400), Image.ANTIALIAS)
    image.save(fp=fileIn)