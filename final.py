from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFilter
import numpy as np


#############################################
'''
Cropping the image
'''



image = Image.open('test-images/a-27.jpg')
cropped_image = image.crop((130, 655, 1470, 2100))
resized_image = cropped_image.resize((850, int((850 / float(cropped_image.size[0])) * float(cropped_image.size[1]))), Image.Resampling.LANCZOS)
gray_image = resized_image.convert("L")
gray_image.save('resized.png')
edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
gaussian_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))
gaussian_image.save('Guass.png')


thresh = 160
binarized_image = gaussian_image.point(lambda p: 255 if p > thresh else 0)
binarized_image.save("output2.png")
edge_image_name = 'edge_detected_image.jpg'
edge_image.save(edge_image_name)


#################################################
def validate_corner(edge_detected_image, initial_point, threshold=10, density_threshold=0.5):
    x, y = initial_point
    horizontal_slice = edge_detected_image[y, x:x+threshold]
    vertical_slice = edge_detected_image[y:y+threshold, x]
    
    horizontal_density = np.mean(horizontal_slice > 128)
    vertical_density = np.mean(vertical_slice > 128)
    
    # Check if the density of edge pixels exceeds the density threshold
    if horizontal_density > density_threshold and vertical_density > density_threshold:
        return True
    return False

def find_valid_corner(edge_detected_image, start_x_range, end_x_range, top, bottom,threshold=20, density_threshold=0.3):
    for x in range(start_x_range, end_x_range):
        for y in range(top, bottom):
            if edge_detected_image[y, x] > 128:  # Detected an edge
                if validate_corner(edge_detected_image, (x, y), threshold, density_threshold):
                    return (x, y)  # Found a valid corner
    return None



#################################################################
#Overlaying red and green points

'''
Column 1 : 
start_x_range, end_x_range = 60, 110   
top, bottom = 10, 35 


Column 2:
start_x_range, end_x_range = 350, 385  
Column 3:
start_x_range, end_x_range =  620, 680
'''



# # Example usage
# edge_detected_np = np.array(Image.open(edge_image_name).convert('L'))
# start_x_range, end_x_range =  620, 680
# top, bottom = 5, 35  
# corner_result = find_valid_corner(edge_detected_np, start_x_range, end_x_range, top, bottom, threshold=20)



# if corner_result is not None:
#     final_x, final_y = corner_result
    
#     print(corner_result)
   
#     trail = Image.open(edge_image_name)  # Ensure this is the correct path to the edge-detected image
#     rgb_image = trail.convert("RGB")
#     draw = ImageDraw.Draw(rgb_image)

#     # Define the radius for the ellipse and the horizontal jump between dots
#     radius = 10
#     horizontal_jump = 38  # Adjusted for the spacing between dots horizontally

#     # Use the dynamically found start_x and start_y for drawing
#     start_x = final_x
#     start_y = final_y

#     # Define other parameters for rows
#     vertical_jump = 30  # Distance between rows
#     rows = 2  # Total number of rows
#     cols = 5   # Assuming 5 options per question

#     # Loop through rows and columns to draw ellipses
#     for row in range(rows):
#         for col in range(cols):
#             # Calculate the center position for each dot
#             center_x = start_x + col * horizontal_jump
#             center_y = start_y + row * vertical_jump

#             # Define the bounding box for the ellipse
#             top_left = (center_x - radius, center_y - radius)
#             bottom_right = (center_x + radius, center_y + radius)

#             # Choose color based on row for visualization
#             color = 'red' if row % 2 == 0 else 'green'

#             # Draw the ellipse
#             draw.ellipse([top_left, bottom_right], outline=color, fill=color)

#     # Save the image with drawn ellipses
#     rgb_image.save("gray_image_with_dots.png")

# else:
#     print("No valid corner found.")
#     # Handle the case where no corner is found appropriately


###########################
'''
Main Logic - col 1
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

# Convert the binarized image to a NumPy array for analysis
binarized_np = np.array(gaussian_image)

trail = Image.open('resized.png')  # Ensure this is the correct path to the edge-detected image
rgb_image = trail.convert("RGB")
draw = ImageDraw.Draw(rgb_image)


start_x_ranges = [(60, 110), (350, 385), (620, 680)]
top, bottom = 5, 35
radius = 10
horizontal_jump = 38
vertical_jump = 30
edge_detected_np = np.array(edge_image)

answer_dictionary = {
    1 : 'A',
    2 : 'B',
    3 : 'C',
    4 : 'D',
    5 : 'E'}
total_ans = []
for index, start_x_range in enumerate(start_x_ranges):
    start_x_range, end_x_range = start_x_range
    corner_result = find_valid_corner(edge_detected_np, start_x_range, end_x_range, top, bottom, threshold=20)
    print(index, corner_result)
    if corner_result:
        initial_x, initial_y = corner_result
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




