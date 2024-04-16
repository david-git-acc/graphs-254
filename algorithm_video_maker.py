import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


# To animate the images we first need to grab them
# we use the standard format of f_0,f_1,f_2,...,f_i,...,f_n to get the images
def get_images(filepath : str) -> list[np.ndarray]:
    
    # We store the images in a simple list, we don't need more complex structures for this
    images : list[np.ndarray] = []
    
    # First we will get all the images in the folder and then store them in the program
    # All frames have a frame number, starting from 0 and incrementing
    i : int = 0
    while True:
        
        this_image_filepath = filepath + f"/f_{i}.png"
        
        if os.path.exists(this_image_filepath):
            
            # Get the next image and add it to our list of images
            next_image = plt.imread(this_image_filepath)
            images.append(next_image)
            
            i += 1
            
        else: break
        
    return images
    

# Create a video 
def make_video(filename : str, filepath : str, fps : int = 1, seconds_per_image : float = 2 ) -> None:
    
    # First we need to actually get hold of the images to animate
    images : list[np.ndarray] = get_images(filepath)
    
    # SETTING UP THE FIGURE so it's the same size as the original images were
    
    # 100 dpi for this computer 
    dpi=100
    height, width, _ = images[0].shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    
    # Determine for how many frames each image should be on the screen for
    frame_count_per_image = ( fps * seconds_per_image )
    
    # The function that will be used to generate each frame of the video
    def animate_video(t):
    
        # Clear the previous elements on the screen to avoid lag
        plt.cla()
            
        # Hide spines, ticks, etc.
        ax.axis('off')
        
        # Get the image to show for this frame
        img = images[int(t / frame_count_per_image)]
        
        # Actually show the image on the plot
        plt.imshow(img)
         
    # Create the animation itself using FuncAnimation    
    anim = FuncAnimation(fig, animate_video, frames = int(frame_count_per_image * len(images)))

    # Save the animation to the filepath with the given filename by the user
    anim.save(filepath + "/" + filename, fps=fps, dpi=dpi)
