import numpy as np
import matplotlib.pyplot as plt

# Rotate the point (x_A,y_A) by angle theta (radians) with respect to the point (x_B,y_B)
def rotate(x_A : float,y_A : float,x_B : float,y_B : float,theta : float) -> tuple[float,float]:
    
    # To rotate the first set of points w.r.t the second, we first need to set them relative to the origin
    # Therefore take the points to rotate and subtract the relative points first
    x_diff = x_A - x_B
    y_diff = y_A - y_B
    
    # Perform the rotation, using the 2D rotational matrix formula
    rotated_x_diff = x_diff * np.cos(theta) - y_diff * np.sin(theta)
    rotated_y_diff = x_diff * np.sin(theta) + y_diff * np.cos(theta)
    
    # Once rotated, we can add the second points back on to get the translated points
    rotated_x = rotated_x_diff + x_B
    rotated_y = rotated_y_diff + y_B
    
    # Give back as a tuple so it's easier to manage
    return (rotated_x, rotated_y)
            
            
# Made a function to create a directed edge arrow as it's quite tedious to do
def directed_edge_arrow(x_A,y_A, x_B, y_B, radius, arrowsize, ax, edgecolour : str = "black" , linestyle : str = "dotted"):
    
    # Get the location differences so the arrows are placed correctly
    # This is calculated mathematically (on paper) and then input into this program
    X_location_diff = ( radius + arrowsize ) * np.cos(np.angle((y_B-y_A)*1j + (x_B-x_A)))
    Y_location_diff = ( radius + arrowsize ) * np.sin(np.angle((y_B-y_A)*1j + (x_B-x_A)))
    
    # The dx and dys will be used to point the arrow in the correct direction - the change in the x and y coords
    dx = x_B - x_A - X_location_diff
    dy = y_B - y_A - Y_location_diff
    
    # Create the arrow for the visual representation of the directed edge
    visual_edge = ax.arrow(x_A, y_A, dx, dy, 
                color=edgecolour, 
                head_length = arrowsize, 
                head_width = arrowsize, 
                linewidth=0.25,
                linestyle = linestyle,
                zorder=0)
    
    return visual_edge


# Given 2 vertices A and B, and a distance midpoint, determine the coordinates of the distance
# midpoint and the coefficients of the quadratic function to simulate a curved edge (for directed edges)

#          --d--
#      ---      ---
#   ---             ---
# A         mid         B
#
# We return a reference to the actual plotted arrow and the midpoint of the arrow
# This is necessary because if we have a di-edge from A to B and another from B to A,they may have different properties
# So we will need to be able to show both of them separately
# The parameter d specifies how far the edges should stretch at the middle, where d = 0 makes them straight lines 
def curved_directed_edge_arrow(sourcev, destv, d : float, ax, edgecolour : str = "black", linestyle : str = "solid") -> tuple:

    # We can create the curved edge by:
    # 1. Finding the point x_C which is on the same x-coordinate as x_A and a rotation of some theta degrees
    # 2. Finding theta via trigonometry
    # 3. Creating the quadratic curve whose height peaks at the vertical distance d and crosses the x-axis at x_A,x_C
    # 4. Taking the points on the curve and rotating them back by 2pi - theta degrees to get the curved edge
    # 5. Creating the arrow for the curved edges by approximation
    # 6  Returning this to the program so we can add it to the graph
      
    # Determine if we want the edge to be going up or down
    edge_sign = 1 if sourcev.name > destv.name else -1
    
    # Get the x and y coordinates
    x_A, y_A = (sourcev.x, sourcev.y)
    x_B, y_B = (destv.x, destv.y)
    
    # Difference between coordinates - used in the trig
    x_diff = x_B - x_A
    y_diff = y_B - y_A
    
    # This is the euclidean distance from x_A that x_C will be
    D = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate the rotation angle
    theta = 2*np.pi - np.arctan(- y_diff / x_diff)
    
    # Determine whether the point should be to the left or the right of x_A
    if x_B > x_A:    
        x_C = x_A + D
    elif x_B < x_A:
        x_C = x_A - D
        
    else:
        
        # If equal x-coordinates then we need to make them vary by the edgesign
        x_C = x_A + edge_sign *  D
        theta *= edge_sign
          
    # This constant will be used in the height function
    # Need a small margin to prevent divide-by-zero errors
    k = edge_sign * 4 * d / ( (x_A - x_C)**2 + 0.0001)
    
    # Creating the height function - a parabola to simulate the directed edge appearance
    height = lambda h : k * (h-x_A) * (h-x_C)
    
    # Get the resolution that we will use from the graph - the number of data points used to build our curve
    res = sourcev.owner.res
    
    # Create the X-axis from x_A to x_C that our parabola will go on
    X = np.linspace(x_A,x_C, res)
    
    # Create the parabolic curve points
    Y = y_A + height(X)
    
    # Now rotate X and Y to the correct position to create the edge
    remapped_X, remapped_Y = rotate(X,Y,x_A,y_A,theta)

    # Get the index of the midpoint
    midpoint_index = len(remapped_X) // 2
    
    # Get the coordinates of midpoint so we can add it
    mid_x = remapped_X[midpoint_index]
    mid_y = remapped_Y[midpoint_index]

    # Get the arrowsize from the graph
    arrowsize = sourcev.owner.arrowsize
    
    # The index of the arrow determines how far along the line it is 
    arrow_index = int( 0.94* (D-sourcev.radius) * res / D )
    
    # Determine the x and y coordinates of the arrow
    arrow_X = remapped_X[arrow_index]
    arrow_Y = remapped_Y[arrow_index]
    
    # Calculate the derivatives of the arrow so we know its direction
    dx = arrow_X - remapped_X[arrow_index - 1] 
    dy = arrow_Y - remapped_Y[arrow_index - 1] 
    
    # Create the arrow itself
    arrow = ax.arrow(arrow_X,arrow_Y,dx,dy,
                     color=edgecolour, 
                     head_length = arrowsize, 
                     head_width = arrowsize, 
                     linewidth=0.25,
                     
                     zorder=0)
    
   
    # The curved line itself, without the arrow
    curved_line = ax.plot(remapped_X, remapped_Y, 
                           color=edgecolour, 
                           linewidth=1, 
                           zorder=0, 
                           clip_on=False,
                           linestyle=linestyle)[0]
    
    
    # Return the references to the curve edge, the arrow and the coordinates of the midpoints
    return (arrow, curved_line, mid_x,mid_y)


# Create a circular self loop arrow at a vertex
# frac_circum is the fraction of diameter of the circle covered by the self loop
#               
#                       -----------------  
#               -------                  -------
#            ---                                ---
#           -                                       -
#         |                                           |
#        -                                             -
#       |                                           _   |    _
#      |          --------------------------          _ |  _
#     |  ---------                          ---------     _  
# -------                   frac_circum * r          -------
#    X1--------------------------------------------------X2
#
# We will cut a chord across the top of the vertex circle, and the points of intersection X1,X2 between the chord and
# the circle will be used to create another circle, which will be used as the curved line of the self loop
# We will then create an arrow for the self loop by calculating its position on the circle so it points directly to the vertex
def selfloop_arrow(sourcev, frac_diam : float, ax, edgecolour : str = "black" , linestyle : str = "solid",
                   background_colour : str = "white"):
    
    # The x and y coordinates of the vertex
    x,y = sourcev.x, sourcev.y
    
    # The radius of the vertex
    r = sourcev.radius
    
    # The chord length, (from X1 to X2), given as a fraction of the diameter
    d : float = frac_diam * 2 * r
    
    # Calculate the angle theta of the vertex, out of 2pi, needed to have a horizontal length of d for the chord length
    theta_chord = np.arccos(1-d**2 / (2 * r**2))
    
    # The angle used to calculate the chord coordinates of the points of intersection between the self loop and vertex circle
    theta = (np.pi - theta_chord) / 2
    
    # Calculate the coordinates of the rightmost point of the chord 
    chord_x = x + r * np.cos(theta)
    chord_y = y + r * np.sin(theta)
    
    # Determine the y-coordinate of the centre of the self-loop circle (calculated on paper and then written here)
    # The centre of the self loop circle will have the same x coordinate so no need to change this
    selfloop_y = 2 * chord_y - y
    
    # Determine the size of the self loop arrow - the graph containing the vertex will have this information stored
    arrowsize = sourcev.owner.arrowsize
    
    # These rotated coordinates will tell us the coordinates to place the arrowhead
    # We do this by rotating the chord the exact angle (arrowsize/r) so that the circle is different from the arrow
    # And then doing it again but for an arbitrarily small change in the angle, giving points very cloes to each other
    # These points will be nearly identical, so their difference should be the derivative which will ensure the arrow
    # points in the correct direction 
    mapped_x1,mapped_y1 = rotate(chord_x, chord_y, x, selfloop_y, arrowsize/r+0.0001)
    mapped_x2,mapped_y2 = rotate(chord_x,chord_y, x, selfloop_y, arrowsize/r)
    
    # This the slefloop, we use a white facecolor to make sure it's hollow to create the self loop appearance
    circ = plt.Circle((x,selfloop_y), r, facecolor=background_colour,edgecolor=edgecolour, zorder=0, clip_on=False, linestyle=linestyle)

    # The circle must be placed on the same axes as the vertex, obviously
    ax.add_patch(circ)
    
    # The arrow itself - we only want the arrow and NOT the edgelength so we can make the difference between the points very 
    # small and then manually determine the size of the arrow through head_length and head_Width
    arrow = ax.arrow(mapped_x1,mapped_y1, 
                      mapped_x2-mapped_x1,
                      mapped_y2-mapped_y1, 
                      head_length = arrowsize, 
                      head_width = arrowsize, 
                      color=edgecolour, zorder=0,
                      clip_on = False)
    
    # We give back the arrowhead and circle so they can be referenced
    # We also give the midpoint coordinates of the arrow so we can set an edge weight for this self loop
    return (arrow, circ, x, selfloop_y + r)


# Get the orientation (closest quadrant/corner) of a 2D point in the axes
def point_orientation(x : float, y : float, aspect_ratio : float = 1):
    
    # Initially we start with 4 possible points - lower left, upper left, lower right, upper right
    # Depending on the aspect ratio we may be able to get centre coordinates as well
    xlocations = ["left", "right"]
    ylocations = ["lower" , "upper"]
    
    # If aspect ratio > 1 then we have more width than height, so horizontal centering is available
    if aspect_ratio > 1:
        xlocations.insert(1, "center")
        
    elif aspect_ratio < 1: 
        
        # Otherwise vice versa, only if it's a square do we not have the ability to centre
        ylocations.insert(1, "center")

    xindex = int( x*len(xlocations)) 
    yindex = int( y*len(ylocations)) 
    
    return ylocations[yindex] + " " + xlocations[xindex]

# Insert newlines in a stream of text to create a paragraph
# Characters per line determines how long we want each line to be before we move onto the next one
# Words will not be interrupted, so if we see a word that is cut in half by the limit we just move the whole word down
def paragraphise(text : str, characters_per_line : int):
    
    # Look at each word by spacing - that's generally the definition of a "word"
    text_words = text.split(" ")
    
    # We will use this as a counter to know when to move
    characters_elapsed = 0
    
    for wordindex, word in enumerate(text_words):
            
        wordlength = len(word)
        
        # We should never encounter a word longer than a full line but if we do, then we quit
        if wordlength > characters_per_line:
            print(f"Word {word} is too long!")
            return 
        
        # If there's already a newline then we can reset
        if "\n" in word: characters_elapsed = 0
        
        # Check if we need to move to the next line - if adding this word would take us over the limit
        elif characters_elapsed + wordlength > characters_per_line:
            
            # If so then add a newline to the previous word to uphold the limit
            text_words[wordindex-1] += "\n"
            
            # Reset the counter of course, otherwise every next word would have a newline
            characters_elapsed = 0
        
        # Increment the counter with the size of the word (+1 because of the implied space)
        characters_elapsed += wordlength + 1
    
    # Don't forget to turn it back into string format
    return " ".join(text_words)