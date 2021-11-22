from icevision.all import *

data_dir = icedata.pennfudan.load_data()
# class_map = ClassMap(['person']) 
parser = icedata.pennfudan.parser(data_dir)
# train_ds, valid_ds = parser.parse()