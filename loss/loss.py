import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# Define the custom loss function
def DistanceBasedLoss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    cortical_class: int = 1,
    trabecular_class: int = 2
) -> torch.Tensor:
    # Ensure shapes match
    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have the same shapes.")

    # Calculate the Dice loss for cortical and trabecular classes
    shape = y.shape[2:]
    y = y.float()
    y_pred = y_pred.float()
    contour = np.zeros(shape, dtype=np.int)
    x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    radius = 12
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    contour[distance < radius] = 1
        
    # Calculate the overextension loss
    trabecular_prob = y_pred[:, trabecular_class, :, :, :]
    cortical_prob = y_pred[:, cortical_class, :, :, :]

    distance_map = distance_transform_edt(1 - cortical_prob.cpu().numpy())

    # Adjust the distance map based on orientation
    adjusted_distance_map = distance_map
    adjusted_distance_map[trabecular_prob > cortical_prob] *= -1
    print(adjusted_distance_map.shape)
    # Visualize the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(contour[32, :, :], cmap="hot")
    axes[0].set_title("Cortical Bone")
    im2 = axes[1].imshow(trabecular_prob[0, 32, :, :], cmap="hot")
    axes[1].set_title("Trabecular Bone")
    im3 = axes[2].imshow(adjusted_distance_map[0, 32, :, :], cmap="RdYlBu", vmin=-np.max(distance_map), vmax=np.max(distance_map))
    axes[2].set_title("Trabecular to Cortical Distance Map")

    plt.show()
    total_loss = torch.sum(torch.relu(-adjusted_distance_map) ** 2)

    return total_loss

batch_size = 1
shape = (64, 64, 64)
data = np.zeros((batch_size, 3, *shape), dtype = np.uint8)
data[0, 1, 10:40, 10:40, 10:40] = 1  # Cortical bone
data[0, 2, 20:40, 20:30, 20:30] = 1  # Trabecular bone
# Simulate predicted data (in practice, this would be the output of your model)
y_pred = torch.tensor(data, dtype=torch.float32)

# Simulate ground truth data (e.g., from annotations)
y = torch.tensor(data, dtype=torch.float32)

print("Label:", y.shape, " Pred:",y_pred.shape)
# Calculate the custom loss
loss = DistanceBasedLoss(y_pred, y, cortical_class=1, trabecular_class=2)
print("Custom Loss:", loss.item())

# Visualize the results