from PIL import Image, ImageDraw, ImageFilter
import sys
import random
import numpy as np

'''
Steps:
1. Load the image
2. Convert it to grayscale
3. Apply Gaussian Noise
4. Binarize the image to differentiate the points easily
5. Find the countours, where will iterate pixel by pixel and check in a predefined area

'''

### Global variables
IMG_WIDTH = 500
IMG_HEIGHT = 500


# Load the image
image = Image.open('test-images/b-27.jpg')
print("Image is %s pixels wide." % image.width)
print("Image is %s pixels high." % image.height)
# resizing the image for ease of analysis
resized_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
# Convert them to grayscale
gray_image = resized_image.convert("L")

# gray_image.save("gray_image.png")

###################################### Applying guassian filter
# This steps helps us to remove the noise if any

# guassain_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1)); # This random kernel value, if the program performs poor, may be we could try adjusting it

# Now we can apply binary thresholding, this to convert the image to black and white, where white dots represent the answer marked which becomes more useful to detect the answer points
thresh = 90
binarized_image = gray_image.point(lambda p: 255 if p > thresh else 0)

binarized_image.save("output2.png")

draw = ImageDraw.Draw(binarized_image)
binarized_image_np = np.array(binarized_image)
##############################################
from PIL import Image, ImageDraw
import numpy as np

# Assuming 'binarized_image' is your image after binarization
binarized_image_np = np.array(binarized_image)

# Convert the binarized image to RGB to draw colored circles
rgb_image = Image.fromarray(binarized_image_np).convert("RGB")
draw = ImageDraw.Draw(rgb_image)

# Define parameters
start_x, start_y = 42, 80
vertical_jump, horizontal_jump = 6, 9
radius = 2
rows, cols = 30, 5  # Assuming 2 rows and 5 columns based on your provided positions

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
##########################################################
'''
Original image with circle to check the overlay of circle
'''
# Assuming 'image' is your original image loaded earlier
# Ensure the image is in RGB mode to allow for drawing in color
rgb_image = gray_image.convert("RGB")
draw = ImageDraw.Draw(rgb_image)

# Define parameters
start_x, start_y = 42, 80
vertical_jump, horizontal_jump = 6, 9
radius = 2
rows, cols = 30, 5  # Assuming 2 rows and 5 columns based on your provided positions

# Iterate over each expected circle position and draw a red circle
for row in range(rows):
    for col in range(cols):
        current_x = start_x + col * horizontal_jump
        current_y = start_y + row * vertical_jump
        # Draw a red ellipse at each position without checking for fill
        draw.ellipse((current_x-radius, current_y-radius, current_x+radius, current_y+radius), outline="red", width=1)

# Save or show the image with red circles
rgb_image.save("original_with_red_circles.png")
# rgb_image.show()


##########################################################
rgb_image = binarized_image.convert("RGB")

# Create a drawing context
draw = ImageDraw.Draw(rgb_image)

# Define the positions for the dots and the radius
positions = [(42, 80), (51, 80), (60, 80), (69, 80), (78, 80)]  # List of tuples for each position
radius = 2

# Draw red dots on the specified positions
for position in positions:
    x, y = position
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='red')

positions = [(42, 86), (51, 86), (60, 86), (69, 86), (78, 86)]  # List of tuples for each position
radius = 2

# Draw red dots on the specified positions
for position in positions:
    x, y = position
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='green')

# Save or show the image with dots
rgb_image.save("gray_image_with_dots.png")  # Save the image



