'''
Flow of the code
1. Import the necessary libraries
2. Convert the image to gray scale image'
3. Resize and crop the image
4. Apply guassian blur
5. Apply binarization for the image
6. Use pillow get image's edges
7. Get the top most left corner for each segment
8. Now use the start index to iterate and find the box which are atleast half filled
'''

##########################################################################################
'''
Import libraries, and preprocess the image
'''
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFilter
import numpy as np


image = Image.open('test-images/b-13.jpg')

cropped_image = image.crop((130, 655, 1470, 2100))
resized_image = cropped_image.resize((850, int((850 / float(cropped_image.size[0])) * float(cropped_image.size[1]))), Image.Resampling.LANCZOS)
gray_scaled_image = resized_image.convert("L")
gray_scaled_image.save('resized.png')
edge_image = gray_scaled_image.filter(ImageFilter.FIND_EDGES)
gaussian_image = gray_scaled_image.filter(ImageFilter.GaussianBlur(radius=1))
gaussian_image.save('guass.png')
guass_np = np.array(gaussian_image)

thresh = 140
binarized_image = gaussian_image.point(lambda p: 255 if p > thresh else 0)
binarized_image.save("output2.png")
edge_image_name = 'edge_detected_image.jpg'
edge_image.save(edge_image_name)
edge_image_np = np.array(edge_image)


##########################################################################################

'''
Corner logic
'''

def validate_corner(edge_detected_image, initial_point, threshold=20, density_threshold=0.5):
    x, y = initial_point
    horizontal_slice = edge_detected_image[y, x:x+threshold]
    vertical_slice = edge_detected_image[y:y+threshold, x]
    
    horizontal_density = np.mean(horizontal_slice > 128)
    vertical_density = np.mean(vertical_slice > 128)
    
    # Check if the density of edge pixels exceeds the density threshold
    if horizontal_density > density_threshold and vertical_density > density_threshold:
        return True
    return False

def find_valid_corner(edge_detected_image , start_x_range, end_x_range, top, bottom,threshold=20, density_threshold=0.5):
    for x in range(start_x_range, end_x_range):
        for y in range(top, bottom):
            if edge_detected_image[y, x] > 128:  # Detected an edge
                if validate_corner(edge_detected_image, (x, y), threshold, density_threshold):
                    return (x, y)  # Found a valid corner
    return None

##########################################################################################
'''
Get the starting corner for each segment
'''
default_start_index = [(100, 25), (375, 25), (650, 25)]
start_values_left_corner = []

start_x_ranges = [(60, 110), (340, 385), (620, 680)]
top, bottom = 5, 35

for index, values in enumerate(start_x_ranges):
    segement_start_x, segement_end_x = values
    # Use edge_image_np here, which is the NumPy array version of edge_image
    ans = find_valid_corner(edge_image_np, segement_start_x, segement_end_x, top, bottom, threshold=20)
    if ans:
        start_values_left_corner.append(ans)
    else:
        # Use default values if no corner is found
        start_values_left_corner.append(default_start_index[index])

print(start_values_left_corner)

##########################################################################################
'''
testing the cornern with dot
'''


# from PIL import Image, ImageDraw

# # Assuming 'trail' is your original or a specific image you want to draw on
# # If 'trail' is not defined, you might need to reload the image
# # trail = Image.open('path_to_your_image.jpg')
# trail = Image.open('resized.png')  # For example, or use any relevant image
# draw = ImageDraw.Draw(trail)

# # Define the radius for the small point/circle
# radius = 3  # Adjust as needed for visibility

# # Loop through your start_values_left_corner to draw points
# for x, y in start_values_left_corner:
#     # Draw a small circle for better visibility
#     # Subtract and add radius to x and y to create a small circle instead of a single point
#     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue')

# # Save or display the modified image
# trail.save("marked_corners.png")  # Save the image with points




##########################################################################################
'''
Check whether the bubble is filled or not
'''
def is_filled_circle(img_array, center, radius, fill_threshold=0.3):
    x_center, y_center = center
    filled_pixels = 0
    total_pixels = 0
    
    for y in range(y_center - radius, y_center + radius + 1):
        for x in range(x_center - radius, x_center + radius + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                if img_array[y, x] < 128:  # Assuming darker pixels are "filled"
                    filled_pixels += 1
                total_pixels += 1
                
    fill_percentage = filled_pixels / total_pixels if total_pixels > 0 else 0
    return fill_percentage > fill_threshold

##########################################################################################
'''
Main Logic iterate through the image, and get the answer options and place an marker on the marked option
'''


binarized_np = np.array(binarized_image)

trail = Image.open('resized.png')   
rgb_image = trail.convert("RGB")
draw = ImageDraw.Draw(rgb_image)



radius = 10
horizontal_jump = 38
vertical_jump = 30


answer_dictionary = {
    1 : 'A',
    2 : 'B',
    3 : 'C',
    4 : 'D',
    5 : 'E'}
total_ans = []
for index, start_x_range in enumerate(start_x_ranges):
    
    
    initial_x, initial_y = start_values_left_corner[index]
    start_x = initial_x + radius
    start_y = initial_y + radius
    segment_wise_ans = []
    for row in range(29):
        row_wise_ans = ''
        if index == 2 and row == 27:
            break
        for col in range(5):  
            
            center_x = start_x + col * horizontal_jump
            center_y = start_y + row * vertical_jump

            if is_filled_circle(binarized_np, (center_x, center_y), radius):
                # Circle is filled more than half, fill with green color
                draw.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill='green')
                row_wise_ans += answer_dictionary[col + 1]
        segment_wise_ans.append(row_wise_ans)
    total_ans.extend(segment_wise_ans)
                
            


# for index, value in enumerate(total_ans):
#     print(index + 1, value)




print(f'Total answer is {len(total_ans)}')  
rgb_image.save("final_output.png")

##########################################################################################

'''
Find the question with with answer marked beside the question

Idea: Since we know the left top corner, we will move in the left till some threshold, and we check whether the intensity is changing3 times are not
'''


# Load the grayscale image
image_path = 'guass.png'  
image = Image.open(image_path).convert('L')  
draw = ImageDraw.Draw(image)  

# Function to analyze the box for handwriting by checking pixel intensity
def analyze_box_for_handwriting(image_array, box_coords, intensity_threshold=128):
    x1, y1, x2, y2 = box_coords
    box_area = image_array[y1:y2, x1:x2]
    handwriting_detected = np.any(box_area < intensity_threshold)
    return handwriting_detected

# Function to check for significant pixels near the right edge of a box
def check_right_edge_for_handwriting(image_array, box_coords, edge_width=2, intensity_threshold=128):
    _, y1, x2, y2 = box_coords  # Use only the relevant coordinates
    right_edge_area = image_array[y1:y2, x2-edge_width:x2]
    edge_detected = np.any(right_edge_area < intensity_threshold)
    return edge_detected

# Initialize parameters for the analysis
start_x, start_y = start_values_left_corner[0]  # Assuming start_values_left_corner is defined
num_rows = 29
distance_between_rows = 30
box_width, box_height = 20, 20

# Convert the image to a NumPy array for pixel analysis
image_array = np.array(image)


for row in range(num_rows):
    current_y = start_y + row * distance_between_rows
    for box_index in range((start_x // box_width) - 1):  
        current_x = box_index * box_width
        box_coords = (current_x, current_y, current_x + box_width, current_y + box_height)

        # Check the current box for handwriting
        if analyze_box_for_handwriting(image_array, box_coords):
            draw.rectangle(box_coords, outline="green", width=2)
            # Check the right edge of the current box for handwriting to decide on highlighting the next box
            if check_right_edge_for_handwriting(image_array, box_coords) and box_index < ((start_x // box_width) - 2):
                next_box_coords = (current_x + box_width, current_y, current_x + 2 * box_width, current_y + box_height)
                draw.rectangle(next_box_coords, outline="green", width=2)


highlighted_image_path = 'highlighted_final_output_adjusted.png'
image.save(highlighted_image_path)
















