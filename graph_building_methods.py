import numpy as np
import matplotlib.pyplot as plt

px = 1/96
res=100

# Import the graph object structure from our graph.py file
from graph import Graph
               
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
# Also stores the weights of each edge so that they can be represented
def arrify_edges(edges : str , weights : list[float]) -> tuple:
    
    # The list that will store all of our bidirectional edges
    bid_edgelist = []
    
    # The list that will store all of our directed edges
    dir_edgelist = []
    
    # Remove all whitespaces/newlines and then split based on commas so we can identify the edges
    edges = edges.replace(" ", "").replace("\n","").split(",")
    
    # Determine how many more edges there are than weights so we can 
    weights_len_diff =  len(edges) - len(weights)
    
    # Make the weights the same size - if no weight is specified then add None so that the weight is not displayed
    weights += [""] * weights_len_diff
    
    # Now we iterate over each edge (hence a step size of 2 as we go over each vertex) 
    # And create the edges
    for i in range(0,len(edges)-1,2):
        substr = edges[i]
        
        # Identify the vertices based on the string structure
        vertexA = substr[1]
        
        # The next vertex will be in the next section (as separated by commas) and the first character
        vertexB = edges[i+1][0]
        
        # Bidirectional edges
        if substr[0] == "{":
            
            # Add the edge and corresponding weight (//2 because each edge is 2 parts)
            bid_edgelist.append((vertexA,vertexB, weights[i // 2]))
            
        # Single direction edges
        elif substr[0] == "(":
            
            # Add the directed edge and its corresponding weight
            dir_edgelist.append((vertexA,vertexB, weights[i // 2]))
            
    # We will use the list of edges to construct the graph
    return (bid_edgelist, dir_edgelist)


# Add gridlines to the plot to understand what's going on
def gridlines(X : np.ndarray,Y : np.ndarray,x : int,y : int) -> None:
    for i in np.linspace(X.min(),X.max(), x+1):
        plt.axvline(i, Y.min(),Y.max(), color="black", zorder=0)
    for j in np.linspace(Y.min(),Y.max(), y+1):
        plt.axhline(j, X.min(), X.max(), color="black", zorder=0)

# Create the grid that the vertices will be based on
# x = number of columns, y = number of rows
# The user can choose if they want the grid to be displayeds
def create_grid(x : int,y : int, heightratio : float, display : bool =False) -> tuple:
    y2 = y * heightratio
    ratio = min(x/y2,y2/x)
    arrs = ( np.linspace( 0.5*(1-ratio),0.5*(1+ratio), res ),  np.linspace(0,1,res) )

    if y2 >= x:
        X,Y = arrs
        # if display:
        #     plt.fill_between(X, [0], [1], color="blue", zorder=0)
    else:
        Y,X = arrs
        # if display:
        #     plt.fill_betweenx(Y, [0],[1], color="blue",zorder=0)

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
def add_vertices(G : Graph, vertexinfo : list[tuple], nrows : int,ncols : int, X : np.ndarray,Y : np.ndarray) -> None:
    
    # Determine width of the graph which will be used to determine the vertex size
    width = X.max() - X.min()
    
    # Radius of each circle
    circrad = width/(2*ncols)
    
    # Drawing each vertex
    for vertex, x, y in vertexinfo:

        # Get the vertex coordinates
        vertex_x, vertex_y = get_coords(x,y,nrows,ncols,X,Y)
        
        G.add_vertex(vertex, vertex_x, vertex_y, circrad)

# Function to plot the edges of our graph
def add_edges(G : Graph, bi_edges : list[tuple],di_edges : list[tuple]):
    
    # Plotting the bidirectional edges is easier
    for vertexA, vertexB, weight in bi_edges:
        
        # Add the edges
        G.add_edge(vertexA,vertexB, both=True, weight=weight)
        
    # Now for the directional edges it's slightly harder
    for vertexA, vertexB, weight in di_edges:
        
        # Add the edge to the graph
        G.add_edge(vertexA,vertexB, both=False, weight=weight)
        
        
# Combine everything together to produce the graph
# The height ratio is the ratio between the y and x components
# Compress determines if we want to compress the schematic first
def create_graph(schematic : str, edges : str, weights : list[float] = [], heightratio : float =2.4,  display : bool =False,
                 vertexcolour : str = "red", edgecolour : str = "black", compress : bool = True,
                 vertex_textcolour : str = "black", edge_textcolour : str = "black" , linestyle : str = "solid",
                 name : str = "G", resolution : int = 1080, background_colour : str = "white", arrowsize : float = None):


    # Create the plot
    fig, ax = plt.subplots(figsize=(resolution*px,resolution*px))
    
    # The plot will always be a square normalised on (0,1) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    # Remove the axes
    if not display: plt.axis("off")

    # Compress the input to eliminate redundant whitespaces
    if compress: schematic = compress_string(schematic)
    
    # Get both types of edges from the user's string input in array format
    bi_edges, di_edges = arrify_edges(edges, weights)
    
    # Get the vertexinfo and number of rows and columns by analysing the string
    vertexinfo,nrows,ncols = string_metainfo(schematic)

    # Create our X-Y grid
    _,_,X,Y = create_grid(ncols,nrows, heightratio, display=display)
    
    if arrowsize is None: arrowsize = 0.8* ( X.max() - X.min() ) / (2 * ncols)
    legendsize = 15
    
    # Create the Graph object
    G = Graph(name, ax, fig=fig, vertexcolour=vertexcolour, edgecolour=edgecolour, arrowsize=arrowsize, 
              vertex_textcolour= vertex_textcolour,edge_textcolour = edge_textcolour,
              aspect_ratio = ncols/ (heightratio*nrows), linestyle=linestyle, legendsize=legendsize,
              background_colour=background_colour)
    
    # Use the information we've gathered to add the vertices to the graph
    add_vertices(G, vertexinfo, nrows,ncols,X,Y)
    
    # Plot the edges onto the graph using all our info
    add_edges(G, bi_edges, di_edges)

    # Return the graph so we can use it
    return G

