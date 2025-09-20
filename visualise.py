import os
import matplotlib.pyplot as plt
from PIL import Image

# Use a raw string or forward slashes for your path
base_folder = r'C:\Appl\fyp\data\cad-cardiac-mri-dataset'

normal_folder_name = 'Normal'
sick_folder_name = 'Sick'

# Create the full path to the image folders
normal_path = os.path.join(base_folder, normal_folder_name)
sick_path = os.path.join(base_folder, sick_folder_name)


# --- NEW, MORE POWERFUL FUNCTION TO FIND THE FIRST IMAGE ---
def find_first_image_deep(root_folder):
    """
    Walks through a directory and its sub-directories to find the first valid image file.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                # Found an image file, return its full path and stop searching
                return os.path.join(dirpath, filename)
    # If the loops finish without finding anything, return None
    return None

# --- Use the new function to find our images ---
print("Searching for images...")
normal_image_file = find_first_image_deep(normal_path)
sick_image_file = find_first_image_deep(sick_path)


# --- Check if we actually found images before trying to open them ---
if normal_image_file and sick_image_file:
    print(f"Found normal image: {normal_image_file}")
    print(f"Found sick image: {sick_image_file}")
    
    # Open the image files
    img_normal = Image.open(normal_image_file)
    img_sick = Image.open(sick_image_file)

    # Display them side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img_normal, cmap='gray')
    ax1.set_title(f'Normal Heart\n(File: {os.path.basename(normal_image_file)})')
    ax1.axis('off')

    ax2.imshow(img_sick, cmap='gray')
    ax2.set_title(f'Sick Heart\n(File: {os.path.basename(sick_image_file)})')
    ax2.axis('off')

    plt.show()
else:
    print("\nError: Still could not find any valid image files.")
    if not normal_image_file:
        print(f" - No images found in: {normal_path}")
    if not sick_image_file:
        print(f" - No images found in: {sick_path}")