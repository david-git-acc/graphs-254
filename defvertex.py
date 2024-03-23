import numpy as np
import matplotlib.pyplot as plt
from drawing_functions import point_orientation
from defedge import Edge

# Defining a vertex
class Vertex():
    def __init__(self, G, name : str, x : float, y : float, radius : float, colour : str = None,
                 textcolour : str = "black"):
        self.owner = G
        self.ax = G.ax
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius  
        self.textcolour = textcolour
        
        # The edges of the vertex map the NAME of the destination vertex to the object vertex
        # E.g if this vertex is called U, and we have an outgoing edge to V, then an entry would be
        # { "V" : (Object Vertex() at 0x823299238238) }
        self.outgoing_edges : dict[str, Vertex] = {}
        self.incoming_edges : dict[str, Vertex] = {}
        self.plotrep : dict = {}
        
        # The colour of the vertex - can be overridden by the user if necessary
        if colour is None:
            self.colour = G.vertexcolour
        else:
            # If the user inputs a colour choice then respect that choice
            self.colour = colour
   
    
    # Change the text colour of a vertex
    def set_textcolour(self, newcolour : str = None) -> None:
        
        # If not specified then we set the default
        if newcolour is None: newcolour = self.owner.textcolour
        
        # We want to be able to easily access the text colour
        self.textcolour = newcolour
        self.plotrep["text"].set_color(self.textcolour)
        

    # Add an annotation to the axes talking about this vertex  
    def annotate(self, text : str, clear_previous : bool = True):
        
        if clear_previous: self.owner.clear_annotations()
        
        # By using the legend
        self.plotrep["visual"].set_label(r"$\mathbf{Vertex\ " + self.name + "}$: " + text)
        
        # Add it to the list of annotations so we can keep track of it
        self.owner.annotations.append(self.plotrep["visual"])

        # We want to make the legend as close to the actual vertex being annotated as possible
        if clear_previous: self.ax.legend(loc=self.quadrant())
        else: self.ax.legend()
        
          
    # Determine the quadrant (upper left, lower right, etc...) that the vertex belongs to
    def quadrant(self) -> str: return point_orientation(self.x, self.y, self.owner.aspect_ratio)
            
    # Highlight the vertex a colour - this is just a wrapper for the graph method
    def highlight(self, colour : str = None, 
    highlight_ratio : float = None, alpha : float = None, highlight_through : bool = False) -> None:
        
        # We just refer to the graph's method instead
        self.owner.highlight_vertex(self.name, colour, 
                                    highlight_ratio=highlight_ratio, alpha=alpha, highlight_through=highlight_through)     
            
    # Change the colour of the vertex
    def set_colour(self, newcolour : str) -> None:
        
        # Remove the existing vertex
        self.plotrep["visual"].remove()
        
        self.colour = newcolour
        
        # Create a circle for the vertex
        circ = plt.Circle((self.x,self.y), self.radius, 
                          facecolor=self.colour,
                          edgecolor="black",
                          zorder=1, 
                          clip_on=False )
        
        # Add the circle
        self.ax.add_patch(circ)
        
        # Update the plot representation with the new circle
        self.plotrep["visual"] = circ
        
    # Remove this vertex's entire visual representation on the graph
    def plot_remove(self) -> None:

        # For everything in the representation, delete ite
        for plotpropname, plotprop in list(self.plotrep.items()):
            
            # Because it's very hard to check if it's already there without creating another reference
            # We will have to just manually try and do nothing if it's already been deleted
            try:
                plotprop.remove()
                
                # Now we will check if it appeared in highlights so we can remove it from there as well
                highlights = self.owner.highlighted_vertices
                
                # If it exists, get rid of it
                if highlights.get(plotpropname) is not None:
                    del highlights[plotpropname]
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
    def add_edge(self, dest, visual_edge, weight : float = None, midpoint : list[float,float] = None, 
                 edgecolour : str = None, textcolour : str = None):
        
        # If no edgecolour selected, let it be the default
        if edgecolour is None: edgecolour = self.owner.edgecolour
        
        # Same for above, there must always be a text colour
        if textcolour is None: textcolour = self.owner.edge_textcolour
        
        # Make sure this edge actually exists
        if not isinstance(dest, Vertex):
            raise Exception(f"Destination vertex for outgoing edge from {self.name} does not exist")
        
        # Instantiate the edge with all the information
        edge = Edge(self, dest, weight, midpoint=midpoint, colour=edgecolour, textcolour=textcolour)
        
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
    
  
    # Get the number of edges entering this vertex
    def indegree(self) -> int: return len(self.incoming_edges)
    
    # Get the number of edges leaving this vertex
    def outdegree(self) -> int: return len(self.outgoing_edges)
        
    # Get the degree of the vertex - bidirectional edges are NOT counted twice
    def degree(self) -> int:
        
        # Get the names of all incoming and outgoing edges so we can identify how many of them there are
        incoming_edge_names : list[str] = list(self.incoming_edges.keys())
        outgoing_edge_names : list[str] = list(self.outgoing_edges.keys())
        
        # If we combine incoming and outgoing and then make them a set we avoid counting the same connection twice
        all_connected_edge_names : set = set( incoming_edge_names + outgoing_edge_names )
        
        # Then the number of connections is just the number of vertices that EITHER 
        # have an incoming or outgoing edge with this vertex
        return len(all_connected_edge_names)