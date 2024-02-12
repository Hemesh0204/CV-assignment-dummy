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
# Apply binarization threshold
thresh = 110
binarized_image = guassain_image.point(lambda p: 255 if p > thresh else 0)
binarized_image.save("output2.png")


####################################################



##############################################
def sobel_edge_detection(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")
    image = ImageOps.expand(image, border=1, fill='black')  # Add a 1-pixel border to handle edge cases
    pixels = np.array(image)

    # Sobel kernels for edge detection
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Prepare arrays for horizontal and vertical gradients
    sx = np.zeros(pixels.shape)
    sy = np.zeros(pixels.shape)

    # Apply Sobel kernels to the image
    rows, cols = pixels.shape
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            region = pixels[row - 1:row + 2, col - 1:col + 2]  # 3x3 region
            sx[row, col] = np.sum(region * Gx)
            sy[row, col] = np.sum(region * Gy)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sx**2 + sy**2)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # Convert back to image
    edge_image = Image.fromarray(np.uint8(gradient_magnitude))
    return edge_image


edge_image = sobel_edge_detection('resized_and_cropped.jpg')

edge_image.save('omr_sheet_edges.jpg')  # Save the result



###############################################
# trial = Image.open('omr_sheet_edges.jpg')
# rgb_image = trial.convert("RGB")

# draw = ImageDraw.Draw(rgb_image)

# # Define parameters for rows
# start_x, end_x = 45, 270  # Assuming fixed width for each row
# start_y = 10  # Starting Y coordinate for the first row
# vertical_jump = 30  # Distance between rows
# row_height = 25  # Height of each rectangle (row)
# rows = 29  # Total number of rows

# # Draw rectangles for each row
# for row in range(rows):
#     top_left = (start_x, start_y + row * vertical_jump)
#     bottom_right = (end_x, start_y + row * vertical_jump + row_height)
#     draw.rectangle([top_left, bottom_right], outline="red")

# # Save or display the image for visual inspection
# rgb_image.save('rectangless.jpg')



###################################

# #Overlaying red and green points

rgb_image = binarized_image.convert("RGB")
draw = ImageDraw.Draw(rgb_image)


radius = 10
horizontal_jump = 38  # Spacing between dots horizontally

# Define parameters for rows
start_x, end_x = 375, 670   # Assuming fixed width for each row
start_y = 25  # Starting Y coordinate for the first row
vertical_jump = 30  # Distance between rows
row_height = 25  # Height of each rectangle (row)
rows = 29  # Total number of rows


# Loop through rows and columns to draw dots
for row in range(29):
    for col in range(5):
        # Calculate current dot's position
        x = start_x + col * horizontal_jump
        y = start_y + row * vertical_jump

        # Choose color based on row (example: alternate between red and green)
        color = 'red' if row % 2 == 0 else 'green'

        # Draw dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

# Save or show the image with dots
rgb_image.save("gray_image_with_dots.png")
# Save the image


#######################################################
'''
Main Logic - col 1
'''

# Assuming 'binarized_image' is your image after binarization
binarized_image_np = np.array(binarized_image)

# Convert the binarized image to RGB to draw colored circles
rgb_image = Image.fromarray(binarized_image_np).convert("RGB")
draw = ImageDraw.Draw(rgb_image)



marked_answers = []

start_array = [100, 375, 650]

# Function to check if a circle is filled more than half
def is_filled_circle(img_array, center, radius):
    x, y = center
    # Extract the square bounding box of the circle
    square_region = img_array[y-radius:y+radius+1, x-radius:x+radius+1]
    # Calculate the filled percentage
    filled_pixels = np.sum(square_region < 128)  # Assuming dark pixels indicate filling
    total_pixels = square_region.size
    filled_percentage = filled_pixels / total_pixels
    return filled_percentage > 0.3  # You may need to adjust this threshold


for index, start_x in enumerate(start_array):

    for row in range(29):
        if index == 2 and row == 28:
            break
        question_answers = []
        for col in range(5):  
            current_x = start_x + col * horizontal_jump
            current_y = start_y + row * vertical_jump
            center = (current_x, current_y)
            
            # Check if the circle at this position is filled more than half
            if is_filled_circle(binarized_image_np, center, radius):
                draw.ellipse((current_x-radius, current_y-radius, current_x+radius, current_y+radius), outline="green")
                question_answers.append(chr(65 + col))  # Convert column number to letter (A-E)
            else:
                draw.ellipse((current_x-radius, current_y-radius, current_x+radius, current_y+radius), outline="red")
        
        if question_answers:  
            marked_answers.append((row + 1 + (29 * index), question_answers))  

    # Save or show the image with circles
    rgb_image.save("marked_circles.png")

    # Print the marked answers
    for question, answers in marked_answers:
        print(f"Question {question}: Options {' '.join(answers)} are marked.")


'''
radius = 10
horizontal_jump = 38  # Spacing between dots horizontally

# Define parameters for rows
start_x, end_x = 100, 270  # Assuming fixed width for each row
start_y = 25  # Starting Y coordinate for the first row
vertical_jump = 30  # Distance between rows
row_height = 25  # Height of each rectangle (row)
rows = 29  # Total number of rows

second column
tart_x, end_x = 375, 670 
start_x, end_x = 650, 870
'''