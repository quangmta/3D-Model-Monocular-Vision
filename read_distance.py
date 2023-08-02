import numpy as np
file_distance_path = "input/25072023-1610/distance.txt"
distance_table = []
with open(file_distance_path,'r') as file:
    for line in file:
        angle,distance = line.split('-')
        distance_table.append([int(angle),float(distance)])
distance_table = np.array(distance_table)
distance_table = distance_table[distance_table[:,0].argsort()]
angle = int(distance_table[0,0])
print(angle)

# folder = 'abc'
# shift = 0
# print('output/{0}/{0}-depth-{1}.npy'.format(folder,str(shift)))