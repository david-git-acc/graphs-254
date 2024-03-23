from drawing_functions import point_orientation, directed_edge_arrow, curved_directed_edge_arrow, selfloop_arrow
import numpy as np


# Defining an edge
# Every vertex has a source and destination vertex, and a midpoint (which may be specified manually if the user wants)
# Also we have an optional weight which can be used
# The curved attribute determines whether this edge is a curved directed edge or not
class Edge():
    def __init__(self, source, destination, weight : float = None, midpoint : list[float,float] = None,
                 curved : bool = False, colour : str = None, textcolour : str = "black"):
        
        # The source and destination vertices must belong to the same graph
        if source.owner is not destination.owner:
            raise Exception(f"Cannot create edge from vertex {source.name} to vertex {destination.name}: they belong to different graphs")
        
        self.owner = source.owner
        self.source = source
        self.ax = source.ax
        self.destination = destination
        self.plotrep : dict = {}
        self.weight = weight
        self.colour = colour
        self.textcolour = textcolour
        self.curved = curved
        
        # Assigning midpoint - if not specified we just take the midpoint of the source and destination verticess
        if midpoint is not None:
            self.midpoint = midpoint
        else:
            # Let it be the linear midpoint so that we know it's specified 
            self.midpoint = [(source.x+destination.x)/2, (source.y+destination.y)/2]
        
        self.colour = colour
        
        # If no colour is specified then let it be the default
        if colour is None: colour = self.owner.edgecolour
        
    # Change the colour of the weight
    def set_textcolour(self, newcolour : str = None):
        
        # If not specified then we set the default
        if newcolour is None: newcolour = self.owner.textcolour
        
        self.textcolour = newcolour
        self.plotrep["text"].set_color(self.textcolour)
        
    # Add an annotation to the axes talking about this edge
    def annotate(self, text : str, clear_previous : bool = True):
        
        if clear_previous: self.owner.clear_annotations()
        
        # By using the legend
        self.plotrep["visual"].set_label(r"$\mathbf{Edge\ " + self.source.name + self.destination.name + "}$: " + text)
        
        # Add it to the list of annotations so we can keep track of it
        self.owner.annotations.append(self.plotrep["visual"])

        # We want to make the legend as close to the actual edge being annotated as possible
        if clear_previous: self.ax.legend(loc=self.quadrant())
        else: self.ax.legend()
        
    # Determine the quadrant (upper left, lower right, etc...) that the edge belongs to, using its midpoint
    def quadrant(self) -> str: return point_orientation(*self.midpoint)    
    
    # Highlight the edge a colour - this is just a wrapper for the graph method
    def highlight(self, colour : str = None, 
    highlight_ratio : float = None, alpha : float = None, highlight_through : bool = False) -> None:
       
        # To use the graph method we need to get the edge name which must be a tuple of values
        edge_name = (self.source.name, self.destination.name)
        
        # Now we can use the graph's method
        self.owner.highlight_edge(edge_name, colour, 
                                    highlight_ratio=highlight_ratio, alpha=alpha, highlight_through=highlight_through) 
            
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
        
        # Check if the other edge from destination to source exists
        other_edge : Edge = self.owner.get_edge(self.destination.name, self.source.name)
        
        # If we have no double arrow, we are a single bidirectional edge then
        # we must update the counterpart edge's weight as well to prevent inconsistency
        if consistent and not self.curved and other_edge is not None and other_edge.weight != val:
            other_edge.set_weight(val)
        
        # If the weight actually exists: otherwise, don't bother (also check if they write none as a string)
        if self.weight is not None and str(self.weight).lower() != "none": 
            
            # Get the radius of the vertex as it is used to determine the fontsize
            radius = self.source.radius
            
            # Place the text on the plot
            weight_text = self.ax.text(*self.midpoint, self.weight, 
                                    fontsize = 150*np.pi*radius,
                                    color=self.textcolour,
                                    zorder=200, 
                                    ha="center",
                                    va="center",
                                    backgroundcolor="white")
            
            # Link the plot representation with this weight text
            self.plotrep.update({ "text" : weight_text })    
    
    # Change the colour of the edge - we need a special method for this because of how complex it can be
    def set_colour(self, newcolour : str ) -> None:
        
        # 3 cases: curved, directed or bidirectional
        # Try to retrieve the other edge. If it doesn't exist, then we know it must be directed
        other_edge : Edge = self.owner.get_edge(self.destination.name, self.source.name)
        
        # This gets rid of the edge's visual representation on the plot so we can add a new version
        self.plotrep["visual"].remove()
        
        self.colour = newcolour
        
        # Check if we are a self loop, as this requires unique code
        if self.source == self.destination:
            
            # Remove the selfloop arrow component as well
            self.plotrep["selfloop_arrow"].remove()
            
            # Generate the new self loop edge 
            new_edge_arrow, new_edge, _, _ = selfloop_arrow(self.source, 0.5, self.ax, 
                                                            self.colour)
            
            # Since the last two cases don't have arrows we do it separately here
            self.plotrep["selfloop_arrow"] = new_edge_arrow
        
        # Create a different type of line depending on the edge type
        # Curved arrows mean we have 2 directed edges in both directions
        elif self.curved:
            
            # If we are a curved arrow then we also need to remove the existing arrow as well
            self.plotrep["arrow"].remove()

            # Create the new edge on the plot
            new_edge_arrow, new_edge,_,_ = curved_directed_edge_arrow(self.source,
                                                self.destination,
                                                self.source.radius * self.owner.curved_edge_stretchiness,
                                                self.ax,
                                                self.colour)
            
            # Since the last two cases don't have arrows we do it separately here
            self.plotrep["arrow"] = new_edge_arrow
            
        # If other edge doesn't exist and NOT curved, then it must be a single direction arrow
        elif other_edge is None:
            new_edge = directed_edge_arrow(self.source.x, self.source.y,
                            self.destination.x, self.destination.y,
                            self.source.radius, self.owner.arrowsize, self.ax
                            , self.colour)    
        
        # If the other edge exists and we're not curved, it's a bidirectional line
        else:           
            # Create the straight line
            new_edge = self.ax.plot([self.source.x,self.destination.x], [self.source.y,self.destination.y], 
                                        linewidth = 1, 
                                        color=self.colour, 
                                        zorder=0, 
                                        clip_on=False)[0]
        
        # Now link this new edge with the visual plot representation of the edge
        self.plotrep["visual"] = new_edge
            
        # If bidirectional (and not a self-loop) don't forget to update the other edge as well
        if not self.curved and other_edge is not None and self.source != self.destination:
            other_edge.plotrep["visual"] = new_edge
            other_edge.colour = newcolour
            
    # Remove the edge from the plot
    def plot_remove(self) -> None:
        
        # For everything in the representation, delete ite
        for plotprop in list(self.plotrep.values()):
            
            # Because it's very hard to check if it's already there without creating another reference
            # We will have to just manually try and do nothing if it's already been deleted
            try:
                plotprop.remove()
            except ValueError:
                pass
        
        # Clear the plot representation, deleting all references to these axes objects
        # and making them eligible for garbage collection
        self.plotrep = {}