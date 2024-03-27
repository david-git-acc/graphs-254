import numpy as np
import matplotlib.pyplot as plt
from helper_functions import point_orientation, paragraphise
from def_edge import Edge

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
   
    # Rename the vertex to some other name
    # The safe keyword checks if a vertex of this name already exists
    # Disabling safe makes this process faster, but renaming a vertex to an existing vertex name is undefined behaviour
    # Only turn "safe" off if you know what you're doing!
    def rename(self, newname : str , safe : bool = True) -> None:
        
        # If the new name is actually the same, then we don't need to do anything        
        if newname == self.name: return
        
        # Only bother checking this if we're in safe mode, since this is O(V)
        if safe:
            # Check that this vertex doesn't already exist in the graph
            for vertexname in self.owner.vertices():
                if vertexname == newname: 
                    raise Exception(f"Vertex {newname} already exists")
                    

        # We need to extract the fontsize from the previous text to stay consistent
        previous_name_fontsize = self.plotrep["text"].get_fontsize()
        
        # Now remove the old text
        self.plotrep["text"].remove()
        
        # We may need to increase or decrease the font size to accommodate the change in the number of characters
        fontsize_ratio = (len(self.name) / len(newname)) ** 0.1
        
        # And replace it with the new text
        self.plotrep["text"] = self.ax.text(self.x,self.y, newname, 
                 fontsize=previous_name_fontsize * fontsize_ratio, 
                 zorder=2,
                 color=self.textcolour,
                 ha="center",
                 va="center")
        
        # REPLACING ALL REFERENCES INVOLVING NAME - VERTICES FIRST
        
        # Replace it with our new name and the same reference to ourselves
        self.owner.V.update({ newname : self })
        
        # Delete the graph's reference to us in the vertices because we will replace it with the new name
        del self.owner.V[self.name]
        
        # NOW FOR EDGES
        
        # Now we need to replace all the outgoing/incoming edge references
        # Start by replacing the incoming edges of all connected vertices - our outgoing edges are their incoming edges
        for outgoing_vertex in list(self.outgoing_edges.values()):
            
            # Delete the old reference and replace it with the new one
            outgoing_vertex.incoming_edges.update({ newname : self })
            del outgoing_vertex.incoming_edges[self.name]
            
            # Now we will replace the vertex in the set of edges of the graph
            # We need to hold a reference to the edge so we can reconnect it to the set of edges with the new name
            this_edge = self.owner.E[(self.name, outgoing_vertex.name)]
            
            # Now we will replace it with the new reference 
            self.owner.E.update({ (newname, outgoing_vertex.name) : this_edge })
            
            # Get rid of the old reference - this does not delete the edge, only the mention in the edge dictionary
            del self.owner.E[(self.name, outgoing_vertex.name)]
        
        # Now for the incoming vertices - again, our incoming vertices are their outgoing vertices
        for incoming_vertex in list(self.incoming_edges.values()):
            
            # Ditto
            incoming_vertex.outgoing_edges.update({ newname : self })
            del incoming_vertex.outgoing_edges[self.name]
            
            # Ditto but reverse, since these edges are incoming to us
            this_edge = self.owner.E[(incoming_vertex.name, self.name)]
            
            # Now we will replace it with the new reference
            self.owner.E.update({ (incoming_vertex.name, newname) : this_edge })
            
            # Get rid of the old reference - this does not delete the edge, only the mention in the edge dictionary
            del self.owner.E[(incoming_vertex.name, self.name)]
        
            
        # UPDATING THE HIGHLIGHTS
        
        # First for the vertex itself:
        vertex_highlight =  self.owner.highlighted_vertices.get(self.name)
        
        # If the vertex highlight actually exists
        if vertex_highlight is not None:
            
            # Create the new reference and delete the old one
            self.owner.highlighted_vertices.update({ newname : vertex_highlight })
            del self.owner.highlighted_vertices[self.name]
        
        # Now for the highlighted edges
        for edge_name, edge_highlight in list(self.owner.highlighted_edges.items()):
            
            # Get the source and destination names of the highlighted edge
            source_name, dest_name = edge_name
            
            # This is what we will be replacing the name of the highlighted edge with in the highlights   
            new_edge_highlight_name = None
            
            # We only want to update edges where this vertex is involved
            if source_name == self.name:
                new_edge_highlight_name = (newname, dest_name)
            elif dest_name == self.name:
                new_edge_highlight_name = (source_name, newname)
            
            if new_edge_highlight_name is not None:
                
                # Add the new reference and delete the old one
                self.owner.highlighted_edges.update({ new_edge_highlight_name : edge_highlight })
                del self.owner.highlighted_edges[(source_name, dest_name)]

        # Finally perform the renaming
        self.name = newname
   
    # Return the list of all vertices (just the edges) directly adjacent to the vertex, whether incoming or outgoing
    # vertices_only determines whether you want JUST the vertex objects themselves or the dictionary of vertexname-vertex pairs
    # if both is set to true, will consider both incoming and outgoing edges
    def connected_vertices(self, vertices_only : bool = True, both : bool = False) -> list:
        
        # We will store ALL the edges here
        edges = {}
        
        # Just add both outgoing and incoming and return
        edges.update(self.outgoing_edges)  
        if both: edges.update(self.incoming_edges)  
        
        # If we only want vertices then just get the values
        if vertices_only: edges = list(edges.values())

        # Otherwise we can just give the full dict as required
        return edges
            
    
    
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
        
        # The actual text string that we will use, but we need to convert it into paragraphs
        textstring = r"$\mathbf{Vertex\ " + self.name + "}$: " + text
        
        # We use the legend to show the text on the figure
        self.plotrep["visual"].set_label(paragraphise(textstring, self.owner.characters_per_line))
        
        # Add it to the list of annotations so we can keep track of it
        self.owner.annotations.append(self.plotrep["visual"])

        self.ax.legend(loc="center", bbox_to_anchor=self.owner.legendloc, fancybox=True, shadow=True, fontsize=self.owner.legendsize)
        
          
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
                 edgecolour : str = None, textcolour : str = None, linestyle : str = None):
        
        # If no edgecolour selected/textcolour/linestyle, let it be the default
        if edgecolour is None: edgecolour = self.owner.edgecolour
        if textcolour is None: textcolour = self.owner.edge_textcolour
        if linestyle is None: linestyle = self.owner.linestyle
        
        # Make sure this edge actually exists
        if not isinstance(dest, Vertex):
            raise Exception(f"Destination vertex for outgoing edge from {self.name} does not exist")
        
        # Instantiate the edge with all the information
        edge = Edge(self, dest, weight, midpoint=midpoint, colour=edgecolour, textcolour=textcolour,linestyle=linestyle)
        
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