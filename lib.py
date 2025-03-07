import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def image_to_surf_2d(image_path, output_path, color_levels=8, epsilon=2.0, min_area=50,
    zFight = False, zIncrement = 0.01):
    """
    Convert an image to a 2D YSFlight SURF file with faces for each color region

    Args:
        image_path: Path to the input image
        output_path: Path to save the SURF file
        color_levels: Number of color levels for reduction
        epsilon: Douglas-Peucker algorithm simplification parameter
        min_area: Minimum area threshold for contours
        zFight : Enable to fix z fighting issue
        zIncrement : Increment value for z axis
    """
    # 1. Load and prepare the image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    height, width = original_img.shape[:2]

    # 2. Reduce colors using k-means clustering
    print("Reducing colors...")
    pixels = original_img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, centers = cv2.kmeans(pixels, color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct the color-reduced image
    reduced_img = centers[labels.flatten()].reshape(original_img.shape).astype(np.uint8)

    # 3. Create labeled image where each pixel has its cluster index
    label_image = labels.reshape(height, width)

    # 4. Find contours for each unique label using skimage
    print("Finding contours using skimage.measure...")
    all_contours = []
    all_colors = []

    for label_value in range(color_levels):
        # Create a binary mask for this label
        binary_mask = (label_image == label_value).astype(np.uint8)

        # Use skimage's find_contours
        raw_contours = measure.find_contours(binary_mask, 0.5)

        # Process each contour
        for raw_contour in raw_contours:
            # Convert skimage contour format to OpenCV format for further processing
            contour = np.array(raw_contour[:, [1, 0]], dtype=np.int32)  # Swap x,y coordinates

            # Reshape for OpenCV functions
            contour = contour.reshape(-1, 1, 2)

            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify contour using Douglas-Peucker algorithm
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Skip if too few points after simplification
            if len(simplified_contour) < 3:
                continue

            # Create a mask for this specific contour
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Extract the mean color from the original image
            mean_color = cv2.mean(original_img, mask=mask)[:3]
            mean_color = tuple(map(int, mean_color))

            all_contours.append(simplified_contour)
            all_colors.append(mean_color)

    # 5. Export to SURF file
    print(f"Found {len(all_contours)} valid contours. Creating SURF file...")

    if zFight:
        # Calculate areas for all contours
        contour_areas = [cv2.contourArea(contour) for contour in all_contours]

        # Create a list of (contour, color, area) tuples
        combined = list(zip(all_contours, all_colors, contour_areas))

        # Sort by area in descending order (largest first)
        combined.sort(key=lambda x: x[2], reverse=True)

        # Unpack the sorted lists
        all_contours, all_colors, _ = zip(*combined)

    export_contours_to_surf(all_contours, all_colors, output_path, zFight, zIncrement)

    # 6. Visualize results
    viz_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    for i, contour in enumerate(all_contours):
        color = all_colors[i]
        # Draw filled contour with the average color
        cv2.drawContours(viz_img, [contour], 0, color, -1)
        # Draw contour outline in black
        cv2.drawContours(viz_img, [contour], 0, (0, 0, 0), 1)

    return original_img, reduced_img, viz_img, len(all_contours)

def export_contours_to_surf(contours, colors, output_path, zFight, zIncrement):
    """
    Export contours as faces in a YSFlight SURF file

    Args:
        contours: List of contours
        colors: List of RGB colors for each contour
        output_path: Path to save the SURF file
        zFight : Enable to fix z fighting issue
        zIncrement : Increment value for z axis
    """
    # Helper functions
    def nl():
        return "\n"

    def ff(value):
        return "{:.6f}".format(float(value))

    # Start SURF file
    buffer = "SURF" + nl()

    # Collect all vertices and create a mapping
    all_vertices = []
    vertex_indices = {}

    cur_z = 0

    """
    for contour in contours:
        for point in contour:
            x, y = point[0]
            z = cur_z

            # Create a hashable vertex key
            vertex_key = (float(x), float(y), float(z))

            if vertex_key not in vertex_indices:
                vertex_indices[vertex_key] = len(all_vertices)
                all_vertices.append(vertex_key)

            if zFight:
                cur_z += zIncrement
    """
    for i, contour in enumerate(contours):
            # Set z-value for this contour
            contour_z = cur_z if zFight else 0.0

            # Process all points in this contour with the same z-value
            for point in contour:
                x, y = point[0]
                z = contour_z  # Use the z-value for this contour

                # Create a hashable vertex key
                vertex_key = (float(x), float(y), float(z))

                if vertex_key not in vertex_indices:
                    vertex_indices[vertex_key] = len(all_vertices)
                    all_vertices.append(vertex_key)

            # Increment z for the next contour if z-fighting prevention is enabled
            if zFight:
                cur_z += zIncrement


    # Write all vertices
    for x, y, z in all_vertices:
        # YSFlight uses different coordinate system
        buffer += f"V {ff(-y)} {ff(z)} {ff(x)} R" + nl()

    # Write each contour as a face

    cur_z = 0

    for i, contour in enumerate(contours):
        buffer += "F" + nl()  # Start face

        # Write color
        r, g, b = colors[i]
        buffer += f"C {r} {g} {b}" + nl()

        # Calculate face center
        center_x = 0
        center_y = 0

        for point in contour:
            center_x += point[0][0]
            center_y += point[0][1]

        center_x /= len(contour)
        center_y /= len(contour)

        contour_z = cur_z if zFight else 0.0
        center_z = contour_z

        # Normal points up (z-axis)
        normal_x = 0
        normal_y = 0
        normal_z = 1

        # Write normal and center
        buffer += f"N {ff(-center_y)} {ff(center_z)} {ff(center_x)} {ff(-normal_y)} {ff(normal_z)} {ff(normal_x)}" + nl()

        # Write vertex indices for this face
        buffer += "V "

        for point in contour:
            x, y = point[0]
            vertex_key = (float(x), float(y), float(contour_z))
            vertex_index = vertex_indices[vertex_key]
            buffer += f"{vertex_index} "

        buffer = buffer.rstrip() + nl()  # Remove trailing space
        buffer += "E" + nl()  # End face

        if zFight:
            cur_z += zIncrement

    # End SURF file
    buffer += "E" + nl()

    # Write to file
    with open(output_path, 'w') as f:
        f.write(buffer)

    print(f"SURF file created at: {output_path}")
    return buffer

def visualize_results(original, reduced, contours):
    """
    Visualize the image processing steps
    """
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(132)
    plt.title(f"Reduced Colors")
    plt.imshow(reduced)
    plt.axis('off')

    plt.subplot(133)
    plt.title("Contour Faces")
    plt.imshow(contours)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    input_image = "th.png"
    output_surf = "model.srf"

    # Process the image
    original, reduced, contours = image_to_surf_2d(
        input_image,
        output_surf,
        color_levels=10,    # Number of colors
        epsilon=5.0,       # Simplification parameter
        min_area=55,        # Minimum contour area
        zFight=True,
        zIncrement=0.01,
    )

    # Visualize the results
    visualize_results(original, reduced, contours)
