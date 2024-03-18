import numpy as np
import matplotlib.pyplot as plt

px = 1/96
res=10

# Defining our graph theory objects
class Graph():
    def __init__(self, name):
        self.name = name
        self.V = []
        self.E = []
        self.adjacency_list = {}
        
    def add_vertices(self, vertexinfo,nrows,ncols,X,Y) -> None:
        
        # Create each vertex and add it to the graph
        for vertex, x, y in vertexinfo:
            
            # Instantiate the vertex
            v = Vertex(vertex, *get_coords(x,y,nrows,ncols,X,Y))
            
            # Add it to the graph
            self.V.append(v)
        
# Defining a vertex
class Vertex():
    def __init__(self,name, x, y):
        self.name = name
        self.x = x
        self.y = y        
        self.edges = []


    
                 
# Compress the string so that any leading whitespaces, either vertically
# or horizontally, are removed
# E.g 
#
#   " 
#      A   B              "A   B  
#                  --->    
#      C   D               C   D"
#             "
# Which tries to maximise the space taken up on the graph by the vertices
# It also adds standardised whitespace
def compress_string(s : str) -> str:
    
    # First split up into array format so easier to identify the redundant spaces
    s_arr = s.split("\n")
    
    # max horizontal length, so we know how far to check for horizontally
    maxlen = max([len(x) for x in s_arr])
    
    # Adding standardised whitespaces everywhere
    for row in range(len(s_arr)):
        
        # Find how many whitespaces we need to add
        diff = maxlen - len(s_arr[row])
        
        # Add this many whitespaces
        s_arr[row] += " " * diff
            
    # Checking from the top of the string downwards for empty whitespaces
    for i in range(len(s_arr)):    
        # The first time we see a horizontal line which isn't trivially full of whitespaces we stop
        # (or we risk damaging the intended structure of the graph)
        if s_arr[i].replace(" ","") != "":
            break
    
    # Same idea here but starting from the bottom of the graph going upwards
    for j in range(len(s_arr)-1, -1, -1):  
        # Same idea
        if s_arr[j].replace(" ","") != "":
            break   
    
    # Now we compress the array vertically be only accepting the indices from i to j
    # As we know any lines outside of this range are just whitespace
    s_arr = s_arr[i:j+1]  
    
    # Now we do the same but horizontally, using the indices a and b
    for a in range(maxlen):
        # This creates the string equivalent to A[:, a] if this were a numpy array 
        # So it selects the ath column in the graph
        newstr = "".join( [x[a] for x in s_arr if len(x) > a ] )
        
        # Again check if it's empty
        if newstr.replace(" ", "") != "":
            break
    
    # Same horizontal column selection but from right to left
    for b in range(maxlen-1, -1, -1):
        newstr = "".join( [x[b] for x in s_arr if len(x) > b] )
        
        # Same idea
        if newstr.replace(" ", "") != "":
            break
    
    # We make a new array form so we can inspect each column and avoid index out of bounds errors
    new_s_arr = []
    
    for row in s_arr:
        # Take the minimum of the indices to avoid an index out of bounds error
        left = min(len(row)-1, a)
        right = min(len(row)-1, b)
        
        # Then add the string from these indices
        new_s_arr.append( row[left:right+1] )  
            
    # Finally return the compressed string, adding the newlines back again
    return "\n".join( new_s_arr ) 
        

# Get all the metainformation about a string to convert it into a graph.
def string_metainfo(s : str) -> tuple:
    s_arr = s.split("\n")
    
    # The number of columns is just the longest horizontal substring
    ncols = max([len(x) for x in s_arr])

    # Number of rows is just the length of the array (hence number of newlines)
    nrows = len(s_arr)

    # Store vertices and their coordinates in this set
    vertexinfo = set([])
    
    # Identifying all the vertices and their coordinates
    for i in range(nrows):
        current_row = s_arr[i]
        
        # Identifying each vertex by discovering its coordinates [i,j]
        for j in range(len(current_row)):
            if current_row[j] != " ": # Anything not a whitespace is a vertex
                vertexinfo.add((s_arr[i][j], j+1, nrows-i))

    return ( vertexinfo, nrows, ncols)


# Given a string that specifies the edges of the graph, turn it into a manageable array format
def arrify_edges(edges : str) -> tuple:
    
    # The list that will store all of our bidirectional edges
    bid_edgelist = []
    
    # The list that will store all of our directed edges
    dir_edgelist = []
    
    # Remove all whitespaces/newlines and then split based on commas so we can identify the edges
    edges = edges.replace(" ", "").replace("\n","").split(",")
    
    # Now we iterate over each edge (hence a step size of 2 as we go over each vertex) 
    # And create the edges
    for i in range(0,len(edges)-1,2):
        substr = edges[i]
        
        # Identify the vertices based on the string structure
        vertexA = substr[1]
        vertexB = edges[i+1][0]
        
        # Bidirectional edges
        if substr[0] == "{":
            
            # Add the edge
            bid_edgelist.append((vertexA,vertexB))
            
        # Single direction edges
        elif substr[0] == "(":
            
            # Add the directed edge
            dir_edgelist.append((vertexA,vertexB))
            
    # We will use the list of edges to construct the graph
    return (bid_edgelist, dir_edgelist)


# Add gridlines to the plot to understand what's going on
def gridlines(X : np.ndarray,Y : np.ndarray,x : int,y : int) -> None:
    for i in np.linspace(X.min(),X.max(), x+1):
        plt.axvline(i, Y.min(),Y.max(), color="black")
    for j in np.linspace(Y.min(),Y.max(), y+1):
        plt.axhline(j, X.min(), X.max(), color="black")

# Create the grid that the vertices will be based on
# x = number of columns, y = number of rows
# The user can choose if they want the grid to be displayeds
def create_grid(x : int,y : int, heightratio : float, display : bool =False) -> tuple:
    y2 = y * heightratio
    ratio = min(x/y2,y2/x)
    arrs = ( np.linspace( 0.5*(1-ratio),0.5*(1+ratio), res ),  np.linspace(0,1,res) )

    if y2 >= x:
        X,Y = arrs
        if display:
            plt.fill_between(X, [0], [1], color="blue")
    else:
        Y,X = arrs
        if display:
            plt.fill_betweenx(Y, [0],[1], color="blue")

    # Determine the width and height of the overall graph figure
    width = X.max()-X.min()
    height = Y.max() - Y.min()

    if display:
        gridlines(X,Y,x,y)
    
    return (width,height,X,Y)

# Given the x and y coordinates, the number of rows and columns, and the X and Y arrays 
# Determine the coordinates on the grid where a vertex should be if placed there
# (computes the centre of the grid cell)
def get_coords(x : float ,y : float, nrows : int,ncols : int, X : np.ndarray,Y : np.ndarray) -> tuple:
    
    # First determine the height and width of the overall grid
    width = X.max()-X.min()
    height = Y.max() - Y.min()

    # Perform the mapping using the variables to determine the grid cell centre of the x/y coordinates
    mapped_x = X.min() + (x-0.5)*width / ncols
    mapped_y = Y.min() + (y-0.5)*height / nrows
    
    # Return the coordinates
    return (mapped_x, mapped_y)


# Plot the vertices on the graph
def plot_vertices(vertexinfo : list[tuple], ax, nrows : int,ncols : int, X : np.ndarray,Y : np.ndarray, slight : float =0.98) -> None:
    
    # Determine width of the graph which will be used to determine the vertex size
    width = X.max() - X.min()
    
    # Radius of each circle
    circrad = slight * width/(2*ncols)
    
    # As the height of the circle = the diameter, fontsize will be proportional to the radius
    fontsize = 50*3*np.pi * circrad
    
    # Drawing each vertex
    for vertex, x, y in vertexinfo:
        
        # Create a circle for each vertex according to its coordinates
        circ = plt.Circle(get_coords(x,y,nrows,ncols,X,Y), 
                          circrad, 
                          facecolor="red",
                          edgecolor="black",
                          zorder=1 )
        
        # Add the circle
        ax.add_patch(circ)
        
        # Write the name of the vertex in the circle, centred in the circle
        plt.text(*get_coords(x,y,nrows,ncols,X,Y),
                 vertex, 
                 fontsize=fontsize, 
                 zorder=2,
                 color="black",
                 ha="center",
                 va="center")

# Function to plot the edges of our graph
def plot_edges(vertexinfo : list[tuple], bi_edges : list[tuple],di_edges : list[tuple], nrows : int, ncols : int, X : np.ndarray , Y : np.ndarray, slight : float, arrowsize : float =0.01):
    
    # Turn it into a dict for programmatically easy and efficient access
    # Also we want to get the graph coordinates whcih requires the number of rows and columns and the X-Y coords
    vertexinfo_dict = { vertex : get_coords(x,y, nrows, ncols, X, Y) for vertex, x, y in vertexinfo }
    
    # Plotting the bidirectional edges is easier
    for vertexA, vertexB in bi_edges:
        
        # Grab their x-y coordinates from the dict
        x_A, y_A = vertexinfo_dict[vertexA]
        x_B, y_B = vertexinfo_dict[vertexB]
        
        # Plot the edge
        plt.plot([x_A,x_B], [y_A,y_B], linewidth = 1, color="black", zorder=0)
        
    # Radius of each circle
    circrad = slight * (X.max() - X.min())/(2*ncols)
    
    # Now for the directional edges it's slightly harder
    for vertexA, vertexB in di_edges:
        
        # Grab their x-y coordinates from the dict, same as before
        x_A, y_A = vertexinfo_dict[vertexA]
        x_B, y_B = vertexinfo_dict[vertexB]
        
        # Get the location differences so the arrows are placed correctly
        # This is calculated mathematically (on paper) and then input into this program
        X_location_diff = ( circrad + arrowsize ) * np.cos(np.angle((y_B-y_A)*1j + (x_B-x_A)))
        Y_location_diff = ( circrad + arrowsize ) * np.sin(np.angle((y_B-y_A)*1j + (x_B-x_A)))
        
        # The dx and dys will be used to point the arrow in the correct direction - the change in the x and y coords
        dx = x_B - x_A - X_location_diff
        dy = y_B - y_A - Y_location_diff
    
        # Draw the directional arrow
        plt.arrow(x_A, y_A, dx, dy, 
                  color="black", 
                  head_length = arrowsize, 
                  head_width = arrowsize, 
                  linewidth=0.25,
                  zorder=0)
         
s = ( 
'''
   A                          D
                     I
   C                      F
          H      
   B                       E   
   
                   J
   G                    K 

''' )



edges = "{ A ,  D } , { C ,  F} , {B,E} , (A,B), (A,E) "

# Combine everything together to produce the graph
# The height ratio is the ratio between the y and x components
def create_graph(schematic : str, edges : str, heightratio : float =2.4, display : bool =False):

    # Create the plot
    fig, ax = plt.subplots(figsize=(1080*px,1080*px))
    
    # Add the slight factor to prevent the circles from clipping with the wall
    slight = 0.98
    
    # The plot will always be a square normalised on (0,1) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    # Remove the axes
    if not display:
        plt.axis("off")

    # Create the Graph object
    G = Graph("G")

    # Compress the input to eliminate redundant whitespaces
    schematic = compress_string(schematic)
    
    # Get both types of edges from the user's string input in array format
    bi_edges, di_edges = arrify_edges(edges)
    

    # Get the vertexinfo and number of rows and columns by analysing the string
    vertexinfo,nrows,ncols = string_metainfo(schematic)

    # Create our X-Y grid
    _,_,X,Y = create_grid(ncols,nrows, heightratio)
    
    # Add the vertices to the graph
    G.add_vertices(vertexinfo, nrows, ncols, X , Y)

    # Use the information we've gathered to add the vertices to the graph representation
    plot_vertices(vertexinfo, ax, nrows,ncols,X,Y)
    

    # Plot the edges onto the graph using all our info
    plot_edges(vertexinfo, bi_edges, di_edges, nrows, ncols, X , Y, slight)

    plt.savefig("test2.png")

create_graph(s, edges)

schematic = compress_string(s)





# Function to determine the current value to multiply the plot by
#              X
#   ========================
#   |          x           |
#   |     ===========      |
#   |     |         |  y   | Y
#   |     |         |      |
#   |     ===========      |
#   |                      |
#   ========================



