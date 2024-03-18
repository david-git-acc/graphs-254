import numpy as np
import matplotlib.pyplot as plt

px = 1/96
res=10

# Defining our graph theory objects
class Graph():
    def __init__(self, name, ax, arrowsize=0.01):
        self.name = name
        
        # V maps the NAME of the vertex to the object vertex
        # E.g for a vertex called V, then we have an entry { "V" : (Object Vertex() at 0x3299239230) }
        self.V : dict[str, Vertex] = {}
        
        # E maps a pair (source name, destination name) to the actual object edge
        # E.g for vertices called U and V, then we have an entry { ("U", "V") : (Object Edge() at 0x2392230230)} 
        self.E : dict[tuple[str,str], Vertex] = {}
        
        # Get a reference to the axis
        self.ax = ax
        
        # Set the arrowsize - the size of the arrows for directed edges
        self.arrowsize = arrowsize
        
    # Create adjacency list for the graph
    def adjacency_list(self):
        
        # Instantiate empty adjacancy list (set)
        adjacency_list = {}
        
        for vertex_name, vertex in list( self.V.items() ):
            
            # Get the names of all vertices adjacent to this vertex
            vertex_edges = list( vertex.edges.keys() )
            
            # Update the adjacency list accordingly
            adjacency_list.update({ vertex_name : vertex_edges })
            
        # Provide the list back
        return adjacency_list
            
      
    # Add an edge to the graph. both is used for bi-directional edges
    # Also consider the weight of the edge so we can add that as well
    def add_edge(self, source_name : str, destination_name : str, both : bool = False, weight : float = None):
        
        # Add the edges using our directed edge function
        self.add_directed_edge( source_name, destination_name, weight)
        
        # If both edges exist then we remove the arrows
        if both: 
            
            # First add the other directed edge
            self.add_directed_edge( destination_name, source_name, weight)
            
            # Get both the edges
            AtoB : Edge = self.E.get((source_name, destination_name))
            BtoA : Edge  = self.E.get((destination_name, source_name))
            
            # Get the source and destination vertices
            sourcev = self.V.get(source_name)
            destv = self.V.get(destination_name)
            
            # Create the straight line
            visual_edge = self.ax.plot([sourcev.x,destv.x], [sourcev.y,destv.y], 
                                        linewidth = 1, color="black", zorder=0)[0]
            
            # Delete the original arrows so we can replace it with the straight line
            AtoB.plot_remove()
            
            # Swap the visual representation with the new edge for the first edge connection
            AtoB.plotrep["visual"] = visual_edge
            
            # Same as above but vice versa
            BtoA.plot_remove()
            BtoA.plotrep["visual"] = visual_edge
            
            # Then re-add the edgeweight
            AtoB.add_edgeweight()
            
    
    # Add a directed edge to the graph - may or may not have an edgeweight attached
    def add_directed_edge(self, source_name : str, destination_name : str , weight : float = None):
        
        # Get the vertices to do the edge-adding
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
        
        # If the edge already exists, ignore
        if self.E.get((source_name, destination_name)) is not None: return

        # Create an arrow axes object for use
        visual_edge = create_directed_edge_arrow(sourcev.x, sourcev.y, destv.x, destv.y, 
                                                    sourcev.radius, self.arrowsize, self.ax)
        
        # Then add the directed edge to the graph structure
        sourcev.add_edge(destv, visual_edge, weight=weight)
        
    
    # Remove an edge from the graph; both states where it should be a bidirectional removal or just one
    def remove_edge(self, source_name, destination_name, both=False):
        
        self.remove_directed_edge(source_name, destination_name)
        
        # If we want to delete both sides then remove from B to A as well
        if both: self.remove_directed_edge(destination_name, source_name)
            
    # Remove an edge from the graph and its representation
    def remove_directed_edge(self, source_name, destination_name ) -> None:
        
        # Get the vertex whose edge will be deleted
        source_vertex : Vertex = self.V.get(source_name)
        
        # Get the edge itself
        the_edge : Edge = self.E.get( (source_name, destination_name) )
        
        # Get the destination vertex 
        dest_vertex : Vertex = the_edge.destination
        
        # Make sure the edge exists
        if the_edge is None: return
        
        # Remove the edge from the plot representation
        the_edge.plot_remove()
        
        # Get the other edge for use later
        other_edge : Edge = self.E.get((destination_name, source_name))
        
        # If it's None then it doesn't exist, but if it isn't None then the edge does exist and must be compensated
        if other_edge is not None:
            print("not none",source_name, destination_name)

            # Create a directed edge to account for this
            visual_edge = create_directed_edge_arrow(dest_vertex.x, dest_vertex.y, 
                                                        source_vertex.x, source_vertex.y, 
                                                        source_vertex.radius, self.arrowsize, 
                                                        self.ax)
            
            # Now change its visual references to be the new visual edge
            other_edge.plotrep.update({"visual" : visual_edge })
            
            # Add its edgeweight back again
            other_edge.add_edgeweight()
            
            print(source_name, destination_name, list(the_edge.plotrep.keys()))
        
        # Delete all references to the edge so it is eligible for garbage collection
        del self.E[(source_name, destination_name)]
        del source_vertex.edges[destination_name]
        
        
    # # Remove a vertex from the graph and its representation
    # def delete_vertex(self, vertex_name):
        
# Defining a vertex
class Vertex():
    def __init__(self, G : Graph, name : str, x : float, y : float, radius : float):
        self.owner = G
        self.ax = G.ax
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius    
        
        # The edges of the vertex map the NAME of the destination vertex to the object vertex
        # E.g if this vertex is called U, and we have an outgoing edge to V, then an entry would be
        # { "V" : (Object Vertex() at 0x823299238238) }
        self.edges : dict = {}
        self.plotrep : dict = {}
    
    # Add a destination edge to our vertex
    # destination is the destination vertex (bug in mypy prevents me from adding type hints)
    # visual_edge is the actual plotted edge on the plot
    # weight is the float weight assigned to the edge
    def add_edge(self, dest, visual_edge, weight : float = None):
        
        # Instantiate the edge
        edge = Edge(self, dest, weight)
        
        # The edge will now have its visual representation linked to it
        edge.plotrep.update({ "visual" : visual_edge })
        
        # Add the weight of the edge
        edge.add_edgeweight()

        # Add the edge to our list of edges
        self.edges.update({ dest.name : dest })
        
        # Add to the set of edges in the graph
        self.owner.E.update({ (self.name, dest.name) : edge })
        

# Defining an edge
class Edge():
    def __init__(self, source : Vertex, destination : Vertex, weight : float = None):
        
        # The source and destination vertices must belong to the same graph
        if source.owner != destination.owner:
            raise Exception(f"Cannot create edge from vertex {source.name} to vertex {destination.name}: they belong to different graphs")
        
        self.owner = source.owner
        self.source = source
        self.ax = source.ax
        self.destination = destination
        self.plotrep : dict = {}
        self.weight = weight
        
    # Add the edge's weight to the graphical representation
    def add_edgeweight(self):
    
        # If the weight actually exists: otherwise, don't bother
        if self.weight is not None:
            
            # Get the source and destination vertices so we know where to place the edgeweight
            source = self.source
            dest = self.destination
            
            print("Adding edgeweight between" , source.name , "and" , dest.name)
            
            # We know the radius of every vertex
            radius = source.radius
        
            # Determine the coordinates of the weight text
            weight_x = (source.x + dest.x)/2
            weight_y = (source.y + dest.y)/2
            
            # Place the text on the plot
            weight_text = plt.text(weight_x, weight_y, self.weight, 
                                    fontsize = 150*np.pi*radius,
                                    color="black",
                                    zorder=200, 
                                    ha="center",
                                    va="center",
                                    backgroundcolor="white")
            
            # Link the plot representation with this weight text
            self.plotrep.update({ "text" : weight_text })    
    
    # Remove the edge from the plot
    def plot_remove(self):
        
        # For everything in the representation, delete ite
        for plotprop in list(self.plotrep.values()):
            
            # Because it's very hard to check if it's already there without creating another reference
            # We will have to just manually try and do nothing if it's already been deleted
            try:
                plotprop.remove()
            except:
                pass
        
        # Clear the plot representation, deleting all references to these axes objects
        # and making them eligible for garbage collection
        self.plotrep = {}
        
    
            
# Made a function to create a directed edge arrow as it's quite tedious to do
def create_directed_edge_arrow(x_A,y_A, x_B, y_B, radius, arrowsize, ax):
    
    # Get the location differences so the arrows are placed correctly
    # This is calculated mathematically (on paper) and then input into this program
    X_location_diff = ( radius + arrowsize ) * np.cos(np.angle((y_B-y_A)*1j + (x_B-x_A)))
    Y_location_diff = ( radius + arrowsize ) * np.sin(np.angle((y_B-y_A)*1j + (x_B-x_A)))
    
    # The dx and dys will be used to point the arrow in the correct direction - the change in the x and y coords
    dx = x_B - x_A - X_location_diff
    dy = y_B - y_A - Y_location_diff
    
    # Create the arrow for the visual representation of the directed edge
    visual_edge = ax.arrow(x_A, y_A, dx, dy, 
                color="black", 
                head_length = arrowsize, 
                head_width = arrowsize, 
                linewidth=0.25,
                zorder=0)
    
    return visual_edge


    
               
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
def add_vertices(G : Graph, vertexinfo : list[tuple], ax, nrows : int,ncols : int, X : np.ndarray,Y : np.ndarray, slight : float =0.98) -> None:
    
    # Determine width of the graph which will be used to determine the vertex size
    width = X.max() - X.min()
    
    # Radius of each circle
    circrad = slight * width/(2*ncols)
    
    # As the height of the circle = the diameter, fontsize will be proportional to the radius
    fontsize = 150*np.pi * circrad
    
    # Drawing each vertex
    for vertex, x, y in vertexinfo:
        
        # Instantiate the vertex
        v = Vertex(G, vertex, *get_coords(x,y,nrows,ncols,X,Y), circrad)
            
        # Create a circle for each vertex according to its coordinates
        circ = plt.Circle(get_coords(x,y,nrows,ncols,X,Y), 
                          circrad, 
                          facecolor="red",
                          edgecolor="black",
                          zorder=1 )
        
        # Add the circle
        ax.add_patch(circ)
        
        # Link the plot representation of the vertex to the vertex itself so we can have control over it
        v.plotrep.update({ "visual" : circ })
        
        # Write the name of the vertex in the circle, centred in the circle
        vertex_text = plt.text(*get_coords(x,y,nrows,ncols,X,Y),
                 vertex, 
                 fontsize=fontsize, 
                 zorder=2,
                 color="black",
                 ha="center",
                 va="center")

        # Also add the text to the vertex representation as well
        v.plotrep.update({"text" : vertex_text})
        
        # Add the vertex to the graph so we can access them via their string representation
        G.V.update({ vertex : v })

# Function to plot the edges of our graph
def add_edges(G : Graph, bi_edges : list[tuple],di_edges : list[tuple]):
    
    # Plotting the bidirectional edges is easier
    for vertexA, vertexB in bi_edges:
        
        # Add the edges
        G.add_edge(vertexA,vertexB, True)
        
    # Now for the directional edges it's slightly harder
    for vertexA, vertexB in di_edges:
        
        # Add the edge to the graph
        G.add_edge(vertexA,vertexB, False)
        
    
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



edges = "{ A ,  D } , { C ,  F} , {B,E} , (A,B), (A,E), {A,J} "

# Combine everything together to produce the graph
# The height ratio is the ratio between the y and x components
# We add the slight factor to prevent the circles from clipping with the wall
def create_graph(schematic : str, edges : str, heightratio : float =2.4, slight: float = 0.98, display : bool =False):

    # Create the plot
    fig, ax = plt.subplots(figsize=(1080*px,1080*px))
    
    # The plot will always be a square normalised on (0,1) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    # Remove the axes
    if not display:
        plt.axis("off")

    # Create the Graph object
    G = Graph("G", ax)

    # Compress the input to eliminate redundant whitespaces
    schematic = compress_string(schematic)
    
    # Get both types of edges from the user's string input in array format
    bi_edges, di_edges = arrify_edges(edges)
    

    # Get the vertexinfo and number of rows and columns by analysing the string
    vertexinfo,nrows,ncols = string_metainfo(schematic)

    # Create our X-Y grid
    _,_,X,Y = create_grid(ncols,nrows, heightratio)
    
    # Use the information we've gathered to add the vertices to the graph
    add_vertices(G, vertexinfo, ax, nrows,ncols,X,Y)
    
    # Plot the edges onto the graph using all our info
    add_edges(G, bi_edges, di_edges)

    plt.savefig("test2.png")
    
    # Return the graph so we can play with it
    return G


G = create_graph(compress_string(s), edges)


print("-----")

G.add_edge("G", "K", True, weight=5)

G.remove_edge("A","B")

G.remove_edge("G","K")
G.remove_edge("K","G")

# G.add_edge("G","K", False)

# G.remove_edge("J","A")



for key, value in G.adjacency_list().items():
    print(key,value)


# G.remove_edge("A","J")





# G.remove_edge("A","J")
# # G.remove_edge("A","D")

# G.remove_edge("A","D" , True)
# G.add_edge("A","D",True)

# G.add_edge("K","E",False)
# # G.remove_edge("K","E",True)
# G.add_edge("K","E",True)

plt.savefig("test3.png")






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



