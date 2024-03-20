import numpy as np
import matplotlib.pyplot as plt

# Defining our graph theory objects
class Graph():
    def __init__(self, name, ax, arrowsize=0.01, res=100):
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
        
        # Determine the resolution of our edges, in number of data inputs
        self.res = res
        
    # Create adjacency list for the graph
    # The reverse flag when set to True will give all the INCOMING edges for the graph rather than outgoing
    def adjacency_list(self,reverse=False):
        
        # Instantiate empty adjacancy list (set)
        adjacency_list = {}
        
        for vertex_name, vertex in list( self.V.items() ):
            
            # If normal, then we give all outgoing edges
            # I decided to put this check in here instead of separately to avoid having to duplicate code
            if not reverse:
            
                # Get the names of all vertices adjacent to this vertex
                vertex_edges = list( vertex.outgoing_edges.keys() )
                
            else:
                # Get the incoming edges instead
                vertex_edges = list( vertex.incoming_edges.keys())
            
            # Update the adjacency list accordingly
            adjacency_list.update({ vertex_name : vertex_edges })
            
        # Provide the list back
        return adjacency_list
            
    # Get an edge from the graph
    # Used so we can mimic encapsulation and avoid having to access internal attributes
    def get_edge(self, source_name : str, destination_name : str):
        
        # Just a wrapper for the dict
        return self.E.get((source_name, destination_name))
    
    # Same idea but for vertices
    def get_vertex(self, vertex_name : str):
        return self.V.get(vertex_name)
      
    # Add an edge to the graph. both is used for bi-directional edges
    # Also consider the weight of the edge so we can add that as well
    def add_edge(self, source_name : str, destination_name : str, both : bool = False, weight : float = None):
        
        # Add the edges using our directed edge function
        self.add_directed_edge( source_name, destination_name, weight)
        
        # Check if the other edge exists
        other_edge : Edge = self.E.get((destination_name, source_name))
        
        # # If we want 2 directed edges but they have the same weight then there's no point in making a curved line
        # # So we will make them the same edge instead so that the graph is not unnecessarily complicated
        # if not both and weight is not None and other_edge is not None and other_edge.weight == weight :
        #     both = True
            
        # Get the source and destination vertices
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
            
        # If both edges exist then we remove the arrows
        if both: 
            
            # First add the other directed edge
            self.add_directed_edge( destination_name, source_name, weight)
            
            # Get both the edges
            AtoB : Edge = self.E.get((source_name, destination_name))
            BtoA : Edge  = self.E.get((destination_name, source_name))
            
            # Create the straight line
            visual_edge = self.ax.plot([sourcev.x,destv.x], [sourcev.y,destv.y], 
                                        linewidth = 1, 
                                        color="black", 
                                        zorder=0, 
                                        clip_on=False)[0]
            
            # We will also update the midpoints of the edges to be linear again
            midpoint = [(sourcev.x + destv.x)/2 , (sourcev.y + destv.y) / 2]
            
            # Update the midpoints            
            AtoB.midpoint = midpoint
            BtoA.midpoint = midpoint
            
            # Since we're adding an edge for both there can only be one weight, which will be what we specified
            AtoB.weight = weight
            BtoA.weight = weight
            
            # Delete the original arrows so we can replace it with the straight line
            AtoB.plot_remove()
            
            # Swap the visual representation with the new edge for the first edge connection
            AtoB.plotrep["visual"] = visual_edge
            
            # Same as above but vice versa
            BtoA.plot_remove()
            BtoA.plotrep["visual"] = visual_edge
            
            # Then re-add the edg eweight
            AtoB.set_weight(AtoB.weight, consistent=False)
            
        # If the reverse edge from B to A doesn't exist, then we don't have to do anything else
        # But if it does exist, then we will have to change its visual representaton (curved arrow) to avoid clipping
        elif other_edge is not None:
            
            # Get hold of the edge we just added
            edge_just_added : Edge = self.E.get((source_name, destination_name))

            # Remove the existing visual edges so we can replace them
            edge_just_added.plot_remove()
            other_edge.plot_remove()
                  
            # Create the curved line and get its midpoint so we can use this as the edge visualisation
            visual_arrow, visual_edge, midpoint_x, midpoint_y = curved_directed_edge_arrow(destv, sourcev, 0.025, self.ax)
            
            # Set the other edge's new midpoint
            other_edge.midpoint = [midpoint_x, midpoint_y]
            
            # Add its visual plot representation back
            other_edge.plotrep["visual"] = visual_edge
            other_edge.plotrep["arrow"] = visual_arrow
            other_edge.set_weight(other_edge.weight, consistent=False)
            
            # Now we will create the new edge to replace the visual representation of the edge we just added
            visual_arrow2, visual_edge2, mid2_x, mid2_y = curved_directed_edge_arrow(sourcev,destv,0.025,self.ax)
            
            # Now update the midpoints, visual representation and edge weight of the edge we've just added and then we're done
            edge_just_added.midpoint = [mid2_x, mid2_y]  
            edge_just_added.plotrep["arrow"] = visual_arrow2     
            edge_just_added.plotrep["visual"] = visual_edge2
            edge_just_added.set_weight(edge_just_added.weight, consistent = False)

            
    # Add a directed edge to the graph - may or may not have an edge weight attached
    def add_directed_edge(self, source_name : str, destination_name : str , weight : float = None, both: bool = False):
        
        # Get the vertices to do the edge-adding
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
        
        # If the edge already exists, ignore
        if self.E.get((source_name, destination_name)) is not None: return

        # Create an arrow axes object for use
        visual_edge = directed_edge_arrow(sourcev.x, sourcev.y, destv.x, destv.y, 
                                                    sourcev.radius, self.arrowsize, self.ax)
            
        # Then add the directed edge to the graph structure
        sourcev.add_edge(destv, visual_edge, weight=weight)


    # Remove an edge from the graph; both states where it should be a bidirectional removal or just one
    def remove_edge(self, source_name : str, destination_name : str, both : bool = False):
        
        self.remove_directed_edge(source_name, destination_name)
        
        # If we want to delete both sides then remove from B to A as well
        if both: self.remove_directed_edge(destination_name, source_name)
            
    # Remove an edge from the graph and its representation
    def remove_directed_edge(self, source_name, destination_name ) -> None:
        
        # Get the vertex whose edge will be deleted
        source_vertex : Vertex = self.V.get(source_name)
        
        # Get the edge itself
        the_edge : Edge = self.E.get( (source_name, destination_name) )
        
        # Make sure the edge exists
        if the_edge is None: return    
        
        # Get the destination vertex 
        dest_vertex : Vertex = the_edge.destination
        
        # Remove the edge from the plot representation
        the_edge.plot_remove()
        
        # Get the other edge for use later
        other_edge : Edge = self.E.get((destination_name, source_name))
        
        # If it's None then it doesn't exist, but if it isn't None then the edge does exist and must be compensated
        if other_edge is not None:
            
            # Remove the other edge's existing plot functionality so we can replace it
            other_edge.plot_remove()

            # Create a directed edge to account for this
            visual_edge = directed_edge_arrow(dest_vertex.x, dest_vertex.y, 
                                              source_vertex.x, source_vertex.y, 
                                              source_vertex.radius, self.arrowsize, 
                                              self.ax)
            
            # Now change its visual references to be the new visual edge
            other_edge.plotrep.update({"visual" : visual_edge })
            
            # Need to update the midpoint of the other edge as well
            other_edge.midpoint = [ (source_vertex.x + dest_vertex.x)/2 , (source_vertex.y + dest_vertex.y) / 2 ] 
                
        # Delete all references to the edge so it is eligible for garbage collection
        del self.E[(source_name, destination_name)]
        del dest_vertex.incoming_edges[source_name]
        del source_vertex.outgoing_edges[destination_name]
        
        # We need to wait until after the vertex references have been deleted before we set the weight again
        if other_edge is not None:
            
            # Add its edge weight back again
            other_edge.set_weight(other_edge.weight, consistent= False)
        
        
    # Remove a vertex from the graph and its representation
    # Also removes all edges connected to it
    def remove_vertex(self, vertex_name : str) -> None:
        
        # Grab the vertex itself
        the_vertex : Vertex = self.V.get(vertex_name)
        
        # Don't try to delete a vertex that doesn't exist
        if the_vertex is not None:
        
            # We will remove every edge, first from the outgoing edges of the vertex
            for edge_name in list(the_vertex.outgoing_edges.keys()):
                
                # Remove ALL edges, both directions, connecting to this edge
                self.remove_edge(vertex_name, edge_name, both=True)
            
            # Now for the incoming edges
            for edge_name in list(the_vertex.incoming_edges.keys()):
                
                # Remove ALL edges, both directions, connecting to this edge
                self.remove_edge(vertex_name, edge_name, both=True)
            
            # Remove all of the vertex's visual representation
            the_vertex.plot_remove()

            # Finally remove the vertex from representation
            del self.V[vertex_name]
    
    # Add a vertex to the graph 
    def add_vertex(self, vertex_name : str, x : float, y : float, radius : float) -> None:
        
        # Instantiate the vertex
        new_vertex = Vertex(self, vertex_name, x, y, radius)
        
        # Create a circle for the vertex
        circ = plt.Circle((x,y), radius, 
                          facecolor="red",
                          edgecolor="black",
                          zorder=1, 
                          clip_on=False )
        
        # Add the circle
        self.ax.add_patch(circ)
        
        # Link the plot representation of the vertex to the vertex itself so we can have control over it
        new_vertex.plotrep.update({ "visual" : circ })
        
        # Write the name of the vertex in the circle, centred in the circle
        vertex_text = plt.text(x,y, vertex_name, 
                 fontsize=150*np.pi*radius, 
                 zorder=2,
                 color="black",
                 ha="center",
                 va="center")

        # Also add the text to the vertex representation as well
        new_vertex.plotrep.update({"text" : vertex_text})
        
        # Add the vertex to the graph so we can access them via their string representation
        self.V.update({ vertex_name : new_vertex })
        
        
        
        
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
        self.outgoing_edges : dict[str, Vertex] = {}
        self.incoming_edges : dict[str, Vertex] = {}
        self.plotrep : dict = {}
        
    # Remove this vertex's entire visual representation on the graph
    def plot_remove(self) -> None:

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
        
    # Add a directed edge to our vertex
    # destination is the destination vertex (bug in mypy prevents me from adding type hints)
    # visual_edge is the actual plotted edge on the plot
    # weight is the float weight assigned to the edge
    # the midpoint can be optionally input 
    def add_edge(self, dest, visual_edge, weight : float = None, midpoint : list[float,float] = None):
        
        # Make sure this edge actually exists
        if not isinstance(dest, Vertex):
            raise Exception(f"Destination vertex for outgoing edge from {self.name} does not exist")
        
        # Instantiate the edge with all the information
        edge = Edge(self, dest, weight, midpoint=midpoint)
        
        # The edge will now have its visual representation linked to it
        edge.plotrep.update({ "visual" : visual_edge })
        
        # Add the weight of the edge
        edge.set_weight(weight, consistent=False)

        # Add the edge to our list of edges
        self.outgoing_edges.update({ dest.name : dest })
        
        # The destination will now have an incoming edge, so add that
        dest.incoming_edges.update({ self.name : self })
        
        # Add to the set of edges in the graph
        self.owner.E.update({ (self.name, dest.name) : edge })
        

# Defining an edge
# Every vertex has a source and destination vertex, and a midpoint (which may be specified manually if the user wants)
# Also we have an optional weight which can be used
class Edge():
    def __init__(self, source : Vertex, destination : Vertex, weight : float = None, midpoint : list[float,float] = None):
        
        # The source and destination vertices must belong to the same graph
        if source.owner is not destination.owner:
            raise Exception(f"Cannot create edge from vertex {source.name} to vertex {destination.name}: they belong to different graphs")
        
        self.owner = source.owner
        self.source = source
        self.ax = source.ax
        self.destination = destination
        self.plotrep : dict = {}
        self.weight = weight
        
        # Assigning midpoint - if not specified we just take the midpoint of the source and destination verticess
        if midpoint is not None:
            self.midpoint = midpoint
        else:
            # Let it be the linear midpoint so that we know it's specified 
            self.midpoint = [(source.x+destination.x)/2, (source.y+destination.y)/2]
            
    # Set the edge's weight and add it to the graphical representation
    # Default value is None, so by not specifying any input you remove the edge weight
    # The consistent boolean will try to ensure consistent edge weights
    def set_weight(self, val : float = None, consistent : bool = True) -> None:
    
        # Delete the existing weight text if it exists
        if self.plotrep.get("text") is not None:
            self.plotrep["text"].remove()
            del self.plotrep["text"]
            
        # Update the weight to the specified value
        self.weight = val
        
        # Check if this edge is a directed edge with a counterpart
        # If we have an arrow as a plot prop, then there must be a counterpart edge from B to A with different weight
        double_arrow = self.plotrep.get("arrow") is not None
        
        # Check if the other edge from destination to source exists
        other_edge : Edge = self.owner.get_edge(self.destination.name, self.source.name)
        
        # If we have no double arrow, we are a single bidirectional edge then
        # we must update the counterpart edge's weight as well to prevent inconsistency
        if consistent and not double_arrow and other_edge is not None and other_edge.weight != val:
            other_edge.set_weight(val)
        
        # If the weight actually exists: otherwise, don't bother
        if self.weight is not None: 
            
            # Get the radius of the vertex as it is used to determine the fontsize
            radius = self.source.radius
            
            # Place the text on the plot
            weight_text = self.ax.text(*self.midpoint, self.weight, 
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
def directed_edge_arrow(x_A,y_A, x_B, y_B, radius, arrowsize, ax):
    
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

# Given 2 vertices A and B, and a distance midpoint, determine the coordinates of the distance
# midpoint and the coefficients of the quadratic function to simulate a curved edge (for directed edges)

#          --d--
#      ---      ---
#   ---             ---
# A         mid         B
#
# We return a reference to the actual plotted arrow and the midpoint of the arrow
# This is necessary because if we have a di-edge from A to B and another from B to A,they may have different properties
# So we will need to be able to show 
def curved_directed_edge_arrow(sourcev : Vertex, destv : Vertex, d : float, ax) -> tuple:

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

    # Rotation formulas for 2D
    rotate_x = lambda x,y,theta : x * np.cos(theta) - y * np.sin(theta)
    rotate_y = lambda x,y,theta : x * np.sin(theta) + y * np.cos(theta)

    # Calculate the rotation angle
    theta = np.arctan(- y_diff / x_diff)
    
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
    
    # To transform them back, first we need to set them relative to (0,0)
    X -= x_A
    Y -= y_A

    # Rotatin mapping
    remapped_X = rotate_x(X,Y,2*np.pi-theta)
    remapped_Y = rotate_y(X,Y, 2*np.pi - theta)
    
    # Then we add the coordinates back on, which has the same effect as rotating them about (x_A, y_A)
    remapped_X += x_A
    remapped_Y += y_A
    
    # Get the index of the midpoint
    midpoint_index = len(remapped_X) // 2
    
    # Get the coordinates of midpoint so we can add it
    mid_x = remapped_X[midpoint_index]
    mid_y = remapped_Y[midpoint_index]

    # Get the arrowsize from the graph
    arrowsize = sourcev.owner.arrowsize
    
    # The index of the arrow determines how far along the line it is 
    arrow_index = int( 0.95* (D-sourcev.radius) * res / D )
    
    # Determine the x and y coordinates of the arrow
    arrow_X = remapped_X[arrow_index]
    arrow_Y = remapped_Y[arrow_index]
    
    # Calculate the derivatives of the arrow so we know its direction
    dx = arrow_X - remapped_X[arrow_index - 1] 
    dy = arrow_Y - remapped_Y[arrow_index - 1] 
    
    # Create the arrow itself
    arrow = ax.arrow(arrow_X,arrow_Y,dx,dy,
                     color="black", 
                     head_length = arrowsize, 
                     head_width = arrowsize, 
                     linewidth=0.25,
                     zorder=0)
    
   
    # The curved line itself, without the arrow
    curved_line = ax.plot(remapped_X, remapped_Y, 
                           color="black", 
                           linewidth=1, 
                           zorder=0, 
                           clip_on=False)[0]
    
    
    # Return the references to the curve edge, the arrow and the coordinates of the midpoints
    return (arrow, curved_line, mid_x,mid_y)