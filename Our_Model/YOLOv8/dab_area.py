import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.measure import label, regionprops
from Case_study import GT_dab_homo

def calculate_dab_area(binary_image):
    data_reloaded = binary_image

    # Label the connected components in the image
    labels_reloaded = label(data_reloaded, connectivity=2)

    # Extract properties of these components
    props_reloaded = regionprops(labels_reloaded)

    # Initialize the list for storing filtered regions and their counts
    filtered_regions_reloaded = []
    region_counts = []

    # Filter regions by area and apply a margin
    for prop in props_reloaded:
        if prop.area > 10:  # Adjust area threshold as needed
            minr, minc, maxr, maxc = prop.bbox
            margin = 10  # Set the margin size
            minr = max(minr - margin, 0)
            minc = max(minc - margin, 0)
            maxr = min(maxr + margin, data_reloaded.shape[0])
            maxc = min(maxc + margin, data_reloaded.shape[1])
            filtered_regions_reloaded.append((minr, minc, maxr, maxc, prop.area))
            # Count the number of '1's within the region
            region_data = data_reloaded[minr:maxr, minc:maxc]
            count_ones = np.count_nonzero(region_data)
            region_counts.append(count_ones)

    # Sort regions to maintain the specified order
    # Sort by x-coordinate (horizontal position) and split into
    left_regions = sorted([r for r in filtered_regions_reloaded if r[1] < data_reloaded.shape[1] / 2], key=lambda x: x[0])
    right_regions = sorted([r for r in filtered_regions_reloaded if r[1] >= data_reloaded.shape[1] / 2], key=lambda x: x[0])

    # Combine left and right regions
    sorted_regions = left_regions + right_regions

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.imshow(data_reloaded, cmap='gray')

    # Plot each region with a bounding box and label
    for i, (minr, minc, maxr, maxc, _) in enumerate(sorted_regions):
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(rect)
        # Label each box with its region number and count of '1's
        ax.text(minc, minr - 5, f'dab{i} (Count: {region_counts[i]})', color='yellow', fontsize=8, ha='left')

    # Add titles and remove axis ticks
    ax.set_title('Adjusted Regions with Bounding Boxes and Labels')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    return region_counts

from after_predict import dab_homo

area_of_GT_DAB = calculate_dab_area(GT_dab_homo)
area_of_predicted_DAB = calculate_dab_area(dab_homo)




data_reloaded = GT_dab_homo

# Label the connected components in the image
labels_reloaded = label(data_reloaded, connectivity=2)

# Extract properties of these components
props_reloaded = regionprops(labels_reloaded)

# Initialize the list for storing filtered regions and their counts
filtered_regions_reloaded = []
region_counts = []

# Filter regions by area and apply a margin
for prop in props_reloaded:
    if prop.area > 10:  # Adjust area threshold as needed
        minr, minc, maxr, maxc = prop.bbox
        margin = 10  # Set the margin size
        minr = max(minr - margin, 0)
        minc = max(minc - margin, 0)
        maxr = min(maxr + margin, data_reloaded.shape[0])
        maxc = min(maxc + margin, data_reloaded.shape[1])
        filtered_regions_reloaded.append((minr, minc, maxr, maxc, prop.area))
        # Count the number of '1's within the region
        region_data = data_reloaded[minr:maxr, minc:maxc]
        count_ones = np.count_nonzero(region_data)
        region_counts.append(count_ones)

# Sort regions to maintain the specified order
# Sort by x-coordinate (horizontal position) and split into
left_regions = sorted([r for r in filtered_regions_reloaded if r[1] < data_reloaded.shape[1] / 2], key=lambda x: x[0])
right_regions = sorted([r for r in filtered_regions_reloaded if r[1] >= data_reloaded.shape[1] / 2], key=lambda x: x[0])

# Combine left and right regions
sorted_regions = left_regions + right_regions

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 12))
ax.imshow(data_reloaded, cmap='gray')

# Plot each region with a bounding box and label
for i, (minr, minc, maxr, maxc, _) in enumerate(sorted_regions):
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', linewidth=2, fill=False)
    ax.add_patch(rect)
    # Label each box with its region number and count of '1's
    ax.text(minc, minr - 5, f'dab{i} (Count: {region_counts[i]})', color='yellow', fontsize=8, ha='left')

# Add titles and remove axis ticks
ax.set_title('Adjusted Regions with Bounding Boxes and Labels')
ax.set_xticks([])
ax.set_yticks([])

plt.show()

