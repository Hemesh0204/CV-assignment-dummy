from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

# Load the original image
image = Image.open('test-images/b-27.jpg')
#############################################
print(image.size)
left = 130
top = 655
right = 1470
# Define a specific bottom value instead of using 'height + 655'
bottom = 2100  # Example, adjust based on your needs

# Crop the image
cropped_image = image.crop((left, top, right, bottom))

# Resize the image, preserving aspect ratio
base_width = 850
w_percent = (base_width / float(cropped_image.size[0]))
h_size = int((float(cropped_image.size[1]) * float(w_percent)))
resized_image = cropped_image.resize((base_width, h_size), Image.Resampling.LANCZOS)

# Save or process the resized image
resized_image.save('resized_and_cropped.jpg')

##############################################

gray_image = resized_image.convert("L")

guassain_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))
# # Apply binarization threshold
# thresh = 110
# binarized_image = guassain_image.point(lambda p: 255 if p > thresh else 0)
# binarized_image.save("output2.png")

##############################################
import numpy as np

def sobel_edge_detection_vectorized(image_array):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Preallocate gradient arrays
    sx = np.zeros_like(image_array, dtype=np.float32)
    sy = np.zeros_like(image_array, dtype=np.float32)

    # Apply Sobel filter (excluding the border pixels)
    sx[1:-1, 1:-1] = np.sum(np.stack([image_array[1:-1, 1:-1] * Gx[1, 1],
                                      image_array[:-2, :-2] * Gx[0, 0], image_array[:-2, 1:-1] * Gx[0, 1], image_array[:-2, 2:] * Gx[0, 2],
                                      image_array[1:-1, :-2] * Gx[1, 0], image_array[1:-1, 2:] * Gx[1, 2],
                                      image_array[2:, :-2] * Gx[2, 0], image_array[2:, 1:-1] * Gx[2, 1], image_array[2:, 2:] * Gx[2, 2]]), axis=0)

    sy[1:-1, 1:-1] = np.sum(np.stack([image_array[1:-1, 1:-1] * Gy[1, 1],
                                      image_array[:-2, :-2] * Gy[0, 0], image_array[:-2, 1:-1] * Gy[0, 1], image_array[:-2, 2:] * Gy[0, 2],
                                      image_array[1:-1, :-2] * Gy[1, 0], image_array[1:-1, 2:] * Gy[1, 2],
                                      image_array[2:, :-2] * Gy[2, 0], image_array[2:, 1:-1] * Gy[2, 1], image_array[2:, 2:] * Gy[2, 2]]), axis=0)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sx**2 + sy**2)
    
    # Normalize to the range 0 to 255
    gradient_magnitude = np.clip((gradient_magnitude / gradient_magnitude.max()) * 255, 0, 255)

    return np.uint8(gradient_magnitude)

# Usage example
guassain_image_np = np.array(guassain_image)  # Assuming guassain_image is already defined
edge_image_np = sobel_edge_detection_vectorized(guassain_image_np)

# Convert back to image for saving
edge_image = Image.fromarray(edge_image_np)
edge_image.save('omr_sheet_edges.jpg')


#############################################



##########################################
import numpy as np

def find_top_left_corner(edge_detected_image, start_x_range, end_x_range, top, bottom):
    for x in range(start_x_range, end_x_range):
        column = edge_detected_image[top:bottom, x]
        edge_start_indices = np.where(column > 128)[0]  # Assuming edge pixels are white
        if edge_start_indices.size > 0:
            y = edge_start_indices[0] + top  # First edge pixel in column
            # Validate if x is within the specified range to ensure correctness
            if start_x_range <= x < end_x_range:
                return x, y
    return None

# Example usage
edge_detected_np = np.array(Image.open('omr_sheet_edges.jpg').convert('L'))
start_x_range, end_x_range = 80, 110  # Example range where you expect to find the first bubble
top, bottom = 15, 35  # Vertical range to focus the search on the first row of bubbles

# Assuming 'edge_detected_np' is your edge-detected image loaded as a NumPy array
top_left_corner = find_top_left_corner(edge_detected_np, start_x_range, end_x_range, top, bottom)
if top_left_corner:
    print(f"Top-left corner found at: {top_left_corner} and the start values are {start_x_range, top}")
else:
    print("No edge found within the specified range.")

################################
#Overlaying red and green points

from PIL import Image, ImageDraw

# Load the edge-detected image and convert it to an RGB image for drawing
trail = Image.open('omr_sheet_edges.jpg')
rgb_image = trail.convert("RGB")
draw = ImageDraw.Draw(rgb_image)

# Define the radius for the ellipse and the horizontal jump between dots
radius = 10
horizontal_jump = 38  # Adjusted for the spacing between dots horizontally

# Dynamically determined starting points (top_left_corner)
start_x = top_left_corner[0]  # Use the dynamically found start_x
start_y = top_left_corner[1]  # Use the dynamically found start_y

# Define other parameters for rows
vertical_jump = 30  # Distance between rows
rows = 29  # Total number of rows
cols = 5   # Assuming 5 options per question

# Loop through rows and columns to draw ellipses
for row in range(rows):
    for col in range(cols):
        # Calculate the center position for each dot
        center_x = start_x + col * horizontal_jump
        center_y = start_y + row * vertical_jump

        # Define the bounding box for the ellipse
        # Note: The ellipse is drawn within a bounding box that defines its outer edges
        top_left = (center_x - radius, center_y - radius)
        bottom_right = (center_x + radius, center_y + radius)

        # Choose color based on row for visualization (example: alternate between red and green)
        color = 'red' if row % 2 == 0 else 'green'

        # Draw the ellipse
        draw.ellipse([top_left, bottom_right], outline=color, fill=color)

# Save or show the image with dots
rgb_image.save("gray_image_with_dots.png")
