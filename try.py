from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

# Load the original image
image = Image.open('test-images/c-33.jpg')
print(image.size)

base_width = 850
#################################

# # Reference point
# ref_point = (120, 150)

# image_rgb = image.convert('RGB')
# draw = ImageDraw.Draw(image_rgb)

# # Now draw the line in red
# draw.line([ref_point, (ref_point[0] + 100, ref_point[1])], fill='red', width=5)

# # Save the modified image
# image_rgb.save('alignment_check.jpg')



#####################################


# def sobel_edge_detection(image_path):
#     # Load image and convert to grayscale
#     image = Image.open(image_path).convert("L")
#     image = ImageOps.expand(image, border=1, fill='black')  # Add a 1-pixel border to handle edge cases
#     pixels = np.array(image)

#     # Sobel kernels for edge detection
#     Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#     # Prepare arrays for horizontal and vertical gradients
#     sx = np.zeros(pixels.shape)
#     sy = np.zeros(pixels.shape)

#     # Apply Sobel kernels to the image
#     rows, cols = pixels.shape
#     for row in range(1, rows - 1):
#         for col in range(1, cols - 1):
#             region = pixels[row - 1:row + 2, col - 1:col + 2]  # 3x3 region
#             sx[row, col] = np.sum(region * Gx)
#             sy[row, col] = np.sum(region * Gy)

#     # Calculate the gradient magnitude
#     gradient_magnitude = np.sqrt(sx**2 + sy**2)
#     gradient_magnitude *= 255.0 / gradient_magnitude.max()

#     # Convert back to image
#     edge_image = Image.fromarray(np.uint8(gradient_magnitude))
#     return edge_image

# # Example usage
# edge_image = sobel_edge_detection('./test-images/a-3.jpg')
# # edge_image.show()  # Display the result
# edge_image.save('omr_sheet_edges.jpg')  # Save the result






#################################

# Calculate the new dimensions to maintain the aspect ratio
original_width, original_height = image.size
w_percent = (base_width / float(original_width))
h_size = int((float(original_height) * float(w_percent)))

# Resize the image using high-quality resampling
resized_image = image.resize((base_width, h_size), Image.LANCZOS)
# resized_image.save("resized_rectangle.png")

# Convert the resized image to grayscale
gray_image = resized_image.convert("L")

guassain_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))
# Apply binarization threshold
thresh = 110
binarized_image = guassain_image.point(lambda p: 255 if p > thresh else 0)
binarized_image.save("output2.png")

# Prepare to draw on the binarized image
draw = ImageDraw.Draw(binarized_image)

# Image dimensions for the grid drawing
width, height = binarized_image.size

# Grid size
grid_size = 25

# Draw vertical lines
for x in range(0, width, grid_size):
    draw.line(((x, 0), (x, height)), fill="black")

# Draw horizontal lines
for y in range(0, height, grid_size):
    draw.line(((0, y), (width, y)), fill="black")

# Save or show the image with the grid
# Important: Save the 'binarized_image', not the original 'image'
binarized_image.save('grided-big-img.jpg')


###############################################
'''
Rectangle logic
'''
# Convert the binarized image to RGB to draw colored rectangles
rgb_image_with_rectangles = image.convert("RGB")
draw = ImageDraw.Draw(rgb_image_with_rectangles)

# Define parameters for rows
start_x, end_x = 223, 545  # Assuming fixed width for each row
start_y = 675  # Starting Y coordinate for the first row
vertical_jump = 50  # Distance between rows
row_height = 35  # Height of each rectangle (row)
rows = 29  # Total number of rows

# Draw rectangles for each row
for row in range(rows):
    top_left = (start_x, start_y + row * vertical_jump)
    bottom_right = (end_x, start_y + row * vertical_jump + row_height)
    draw.rectangle([top_left, bottom_right], outline="red")

# Save or show the image with rectangles
rgb_image_with_rectangles.save('omr_sheet_with_rectangles.jpg')

print(binarized_image.size)

###########################################

'''
Overlaying point logic
'''


# Assuming 'binarized_image' is already defined in your code
rgb_image = binarized_image.convert("RGB")
draw = ImageDraw.Draw(rgb_image)

# Initial positions and sizes
start_x, start_y = 143, 345  # Starting position for the first dot
radius = 6
horizontal_spacing = 30  # Spacing between dots horizontally
vertical_spacing = 25  # Spacing between rows
cols = 5  # Number of columns (dots in a row)
rows = 29  # Number of rows

# Loop through rows and columns to draw dots
for row in range(rows):
    for col in range(cols):
        # Calculate current dot's position
        x = start_x + col * horizontal_spacing
        y = start_y + row * vertical_spacing

        # Choose color based on row (example: alternate between red and green)
        color = 'red' if row % 2 == 0 else 'green'

        # Draw dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

# Save or show the image with dots
rgb_image.save("gray_image_with_dots.png")
 # Save the image


#######################################
'''
Main Logic
'''
# Assuming 'binarized_image' is your image after binarization
binarized_image_np = np.array(binarized_image)

# Convert the binarized image to RGB to draw colored circles
rgb_image = Image.fromarray(binarized_image_np).convert("RGB")
draw = ImageDraw.Draw(rgb_image)

# Define parameters
start_x, start_y = 143, 345
vertical_jump, horizontal_jump = 25, 28
radius = 12
rows, cols = 29, 5  # Assuming 2 rows and 5 columns based on your provided positions

# Function to check if a circle is filled more than half
def is_filled_circle(img_array, center, radius):
    x, y = center
    # Extract the square bounding box of the circle
    square_region = img_array[y-radius:y+radius+1, x-radius:x+radius+1]
    # Calculate the filled percentage
    filled_pixels = np.sum(square_region < 128)  # Assuming dark pixels indicate filling
    total_pixels = square_region.size
    filled_percentage = filled_pixels / total_pixels
    return filled_percentage > 0.3

# Iterate over each expected circle position
for row in range(rows):
    for col in range(cols): 
        current_x = start_x + col * horizontal_jump
        current_y = start_y + row * vertical_jump
        center = (current_x, current_y)
        # Check if the circle at this position is filled more than half
        if is_filled_circle(binarized_image_np, center, radius):
            print(f"Filled Circle: Row {row + 1}, Column {col + 1}")
            draw.ellipse((current_x-radius, current_y-radius, current_x+radius, current_y+radius), outline="green")
        else:
            draw.ellipse((current_x-radius, current_y-radius, current_x+radius, current_y+radius), outline="red")

# Save or show the image with circles
rgb_image.save("marked_circles.png")



##########################
