import numpy as np

def calculate_volume(label_scan):
    return np.sum(label_scan == 1)

def calculate_intersection_volume(label_scan1, label_scan2):
    intersection = np.logical_and(label_scan1, label_scan2)
    return calculate_volume(intersection)

def calculate_loss(volume1, volume_intersect):
    epsilon = 1e-5
    return (volume1 - volume_intersect) / (volume1 + epsilon)

# Example usage
#  completely disjoinct
label_scan1 = np.zeros((10, 10, 10))
label_scan2 = np.zeros((10, 10, 10))
label_scan1[1:2, 1:2, 1:2] = 1
label_scan2[4:8, 4:8, 4:8] = 1
#FILL INIDE SCAN 2
volume1 = calculate_volume(label_scan1)
volume_intersect = calculate_intersection_volume(label_scan1, label_scan2)
loss = calculate_loss(volume1, volume_intersect)
print("Volume of scan 1:", volume1)
print("Volume of intersection:", volume_intersect)
print("Loss:", loss)

#  Partially intersected
label_scan1 = np.zeros((10, 10, 10))
label_scan2 = np.zeros((10, 10, 10))
label_scan1[2:5, 2:5, 2:5] = 1
label_scan2[3:7, 3:7, 3:7] = 1
volume1 = calculate_volume(label_scan1)
volume_intersect = calculate_intersection_volume(label_scan1, label_scan2)
loss = calculate_loss(volume1, volume_intersect)
print("Volume of scan 1:", volume1)
print("Volume of intersection:", volume_intersect)
print("Loss:", loss)

#  completely intersected
label_scan1 = np.zeros((10, 10, 10))
label_scan2 = np.zeros((10, 10, 10))
label_scan1[3:7, 3:7, 3:7] = 1
label_scan2[1:9, 1:9, 1:9] = 1
volume1 = calculate_volume(label_scan1)
volume_intersect = calculate_intersection_volume(label_scan1, label_scan2)
loss = calculate_loss(volume1, volume_intersect)
print("Volume of scan 1:", volume1)
print("Volume of intersection:", volume_intersect)
print("Loss:", loss)