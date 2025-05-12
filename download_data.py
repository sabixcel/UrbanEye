
from roboflow import Roboflow

rf = Roboflow(api_key="JfOzwbUmlZk6Y0ScivKx")
#project = rf.workspace("myspace-0sj0p").project("final-garbage-detection-kxcnh") #garbage
#project = rf.workspace("myspace-0sj0p").project("vandalisme-grafiti-25p7w") #graffiti
#project = rf.workspace("myspace-0sj0p").project("road-shield4-yhtot") #
#project = rf.workspace("myspace-0sj0p").project("pole-defect-new-ecsgd") # 
#project = rf.workspace("myspace-0sj0p").project("power_bird-c0jte") #craws on wires
#project = rf.workspace("myspace-0sj0p").project("cracks-oln9b-wnwzp") #
#project = rf.workspace("myspace-0sj0p").project("rer-anvdb-2zskx") #
#project = rf.workspace("myspace-0sj0p").project("bear-hdgqp-6vnjc") #bears
#project = rf.workspace("myspace-0sj0p").project("bencana-vdz5p") #floods
project = rf.workspace("myspace-0sj0p").project("tgreeb-zvzps") #illegal parking
dataset = project.version(1).download("yolov11")
