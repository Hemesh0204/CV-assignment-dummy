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


image = Image.open('test-images/c-33.jpg')

cropped_image = image.crop((130, 655, 1470, 2100))
resized_image = cropped_image.resize((850, int((850 / float(cropped_image.size[0])) * float(cropped_image.size[1]))), Image.Resampling.LANCZOS)
gray_scaled_image = resized_image.convert("L")
gray_scaled_image.save('resized.png')
edge_image = gray_scaled_image.filter(ImageFilter.FIND_EDGES)
gaussian_image = gray_scaled_image.filter(ImageFilter.GaussianBlur(radius=1))
gaussian_image.save('Guass.png')


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
                
            


for index, value in enumerate(total_ans):
    print(index + 1, value)




print(f'Total answer is {len(total_ans)}')  
rgb_image.save("final_output_with_filled_circles.png")

##########################################################################################

