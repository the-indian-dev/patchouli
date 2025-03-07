import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def image_to_surf_2d(image_path, output_path, color_levels=8, epsilon=2.0, min_area=50):
    """
    Convert an image to a 2D YSFlight SURF file with faces for each color region

    Args:
        image_path: Path to the input image
        output_path: Path to save the SURF file
        color_levels: Number of color levels for reduction
        epsilon: Douglas-Peucker algorithm simplification parameter
    """
    # 1. Load and prepare the image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    height, width = original_img.shape[:2]

    # 2. Reduce colors using k-means clustering
    print("Reducing colors...")
    pixels = original_img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct the color-reduced image
    reduced_img = centers[labels.flatten()].reshape(original_img.shape).astype(np.uint8)

    # Convert to grayscale for contour detection
    gray_img = cv2.cvtColor(reduced_img, cv2.COLOR_RGB2GRAY)

    # 3. Find contours for each color region
    print("Finding contours...")
    contours = []
    colors = []

    # Instead of using different levels, we'll use binary thresholding to extract each color region
    for color_idx in range(color_levels):
        # Create a mask for this color
        color_mask = (labels.reshape(-1) == color_idx).reshape(height, width).astype(np.uint8) * 255

        # Find contours in this mask
        # Using OpenCV's findContours directly
        opencv_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour in this color region
        for contour in opencv_contours:
            # Only process contours with sufficient area (avoid tiny noise)
            if cv2.contourArea(contour) > min_area:  # Minimum area threshold
                # 4. Apply RDP algorithm to simplify the contour
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

                # Skip if too few points after simplification
                if len(simplified_contour) >= 3:
                    # Get the average color for this region
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)  # Fill the contour

                    # Get the average color of the region in the original image
                    mean_color = cv2.mean(original_img, mask=mask)[:3]  # RGB
                    mean_color = tuple(map(int, mean_color))  # Convert to integers

                    contours.append(simplified_contour)
                    colors.append(mean_color)

    # 5. Export to SURF file
    print(f"Found {len(contours)} valid contours. Creating SURF file...")
    export_contours_to_surf(contours, colors, output_path)

    # Visualize results
    viz_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    for i, contour in enumerate(contours):
        color = colors[i]
        # Draw filled contour with the average color
        cv2.drawContours(viz_img, [contour], 0, color, -1)
        # Draw contour outline in black
        cv2.drawContours(viz_img, [contour], 0, (0, 0, 0), 1)

    return original_img, reduced_img, viz_img, len(contours)

def export_contours_to_surf(contours, colors, output_path):
    """
    Export contours as faces in a YSFlight SURF file

    Args:
        contours: List of contours
        colors: List of RGB colors for each contour
        output_path: Path to save the SURF file
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

    for contour in contours:
        for point in contour:
            x, y = point[0]
            z = 0.0  # 2D, so z=0

            # Create a hashable vertex key
            vertex_key = (float(x), float(y), float(z))

            if vertex_key not in vertex_indices:
                vertex_indices[vertex_key] = len(all_vertices)
                all_vertices.append(vertex_key)

    # Write all vertices
    for x, y, z in all_vertices:
        # YSFlight uses different coordinate system
        buffer += f"V {ff(-y)} {ff(z)} {ff(x)} R" + nl()

    # Write each contour as a face
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
        center_z = 0  # 2D plane

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
            vertex_key = (float(x), float(y), 0.0)
            vertex_index = vertex_indices[vertex_key]
            buffer += f"{vertex_index} "

        buffer = buffer.rstrip() + nl()  # Remove trailing space
        buffer += "E" + nl()  # End face

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
    input_image = "image.jpg"
    output_surf = "model.surf"

    # Process the image
    original, reduced, contours = image_to_surf_2d(
        input_image,
        output_surf,
        color_levels=8,    # Number of colors
        epsilon=1.0,        # Simplification parameter
        min_area=50        # Minimum contour area
    )

    # Visualize the results
    visualize_results(original, reduced, contours)
