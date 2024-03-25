import numpy as np
import matplotlib.pyplot as plt
from helper_functions import directed_edge_arrow, curved_directed_edge_arrow, selfloop_arrow, point_orientation, paragraphise
from def_edge import Edge
from def_vertex import Vertex

# Defining our graph theory objects
class Graph():
    def __init__(self, name, ax, fig = plt.gcf(), arrowsize : float =0.01, res : int =100, vertexcolour : str ="red",edgecolour : str="black",
                 vertex_textcolour : str = "black", edge_textcolour : str = "black", curved_edge_stretchiness : float = 1.4,
                 aspect_ratio : float = None, linestyle : str = "dotted", legendloc : tuple = (0.5, -0.025), legendsize : float = 20):
        
        self.name = name
        
        # V maps the NAME of the vertex to the object vertex
        # E.g for a vertex called V, then we have an entry { "V" : (Object Vertex() at 0x3299239230) }
        self.V : dict[str, Vertex] = {}
        
        # E maps a pair (source name, destination name) to the actual object edge
        # E.g for vertices called U and V, then we have an entry { ("U", "V") : (Object Edge() at 0x2392230230)} 
        self.E : dict[tuple[str,str], Edge] = {}
        
        # Store all highlighted vertices and edges
        # The user will be able to highlight vertices and edges in the colour they want
        self.highlighted_vertices : dict = {}
        self.highlighted_edges : dict = {}
        
        # This ratio determines how heavy the highlighting on a vertex will be
        self.highlight_ratio_vertices = 1.4
        
        # Ditto, but for edges
        self.highlight_ratio_edges = 10
        
        # This value states how much transparency the highlighted regions will have - 0 is invisible, 1 is opaque
        self.highlight_alpha = 0.5 
        
        # Determines if highlighting will be done OVER the vertices (True) and edges or underneath (False)
        self.highlight_through = False
        
        # Get a reference to the axis and figure objects
        self.ax = ax
        self.fig = fig
        
        # MISCELLANEOUS ATTRIBUTES - used in graph customisation
        
        # Determine the default colour of our vertices and edges
        self.vertexcolour = vertexcolour
        self.edgecolour = edgecolour
        
        # The textcolour for all the weights and vertices
        self.vertex_textcolour = vertex_textcolour
        self.edge_textcolour = edge_textcolour
        
        # Set the arrowsize - the size of the arrows for directed edges
        self.arrowsize = arrowsize
        
        self.linestyle = linestyle
        
        # This constant determines how much curved edges will be stretched out
        self.curved_edge_stretchiness = curved_edge_stretchiness
        
        # Determine the resolution of our edges, in number of data inputs
        self.res = res
        
        # We need to keep track of all annotations in the graph
        self.annotations : list = []
        
        # The coordinates and size of the legend, which is used for annotations
        self.legendloc = legendloc
        self.legendsize = legendsize
        
        # Get the aspect ratio of the graph
        self.aspect_ratio = aspect_ratio
        
        # How many characters on a single line of annotation are allowed
        # 96 is the number of pixels in an inch, [0] gets the WIDTH of the figure
        self.characters_per_line = int( fig.get_size_inches()[0] * 10 )
    
        # This isn't actually a plot, it's just so that I can add annotations for the graph
        # Make the plot go outside of [0,1] so we can't see it in the graph but we can still reference it
        self.legend_ref = ax.scatter([10,10],[20,20], marker=f"${self.name}$", color=self.vertex_textcolour)
    
    
    # Add an annotation to the axes
    def annotate(self, text : str, clear_previous : bool = True):
        
        if clear_previous: self.clear_annotations()
        
        # We use the legend to show the text on the figure
        self.legend_ref.set_label(paragraphise(text, self.characters_per_line))
        
        # Add it to the list of annotations so we can keep track of it
        self.annotations.append(self.legend_ref)

        self.ax.legend(loc="center", bbox_to_anchor=self.legendloc, fancybox=True, shadow=True, fontsize=self.legendsize)
        
        
    # Get the edges of the graph - if objects = True then give only the objects, otherwise give only the keys
    def edges(self, objects : bool = False) -> list[Edge]: 
        
        if objects: return list(self.E.values())
        else: return list(self.E.keys())
    
    # Same as above but for vertices
    def vertices(self, objects : bool = False) -> list[Vertex]: 
        
        if objects: return list(self.V.values())
        else: return list(self.V.keys())
    
    # Save the graph as an image
    def save(self, name : str = None) -> None: 
        
        # If they don't specify, just save it as the name of the graph
        if name is None: name = self.name + ".png"
        
        
        plt.savefig(name)
        
    # Set the text colour of every vertex in the graph to a given colour
    def set_vertex_textcolour(self, colour : str) -> None:
        
        # Do for each vertex, set its colour
        for vertex in list(self.V.keys()):
            self.get_vertex(vertex).set_textcolour(colour)
    
    # Set the text colour of every weight for every edge in the graph to a given colour
    def set_edge_textcolour(self, colour : str) -> None:
        
        # For each edge set its colour
        for edge in list(self.E.keys()):
            self.get_edge(*edge).set_textcolour(colour)
            
    # Remove all the annotations about the graph
    def clear_annotations(self) -> None:
        
        # Destroy the labels
        for annotation in self.annotations:
            annotation.set_label(None)
        
        # Destroy the references
        self.annotations = []
        
        # Finally destroy the legend
        try: self.ax.get_legend().remove()
        except: pass    

    # Set all vertices in the graph to a given colour.
    def set_vertex_colour(self, colour : str) -> None:
        
        # Get each vertex and assign it the same colour - we will create a mapping here
        vertex_list : list[str] = list(self.V.keys())
        colour_list : list[str] = [colour] * len(vertex_list)
        
        # This will be the assignment of colours (all the same colour) to assign to each vertex
        colour_assignment : dict[str,str] = dict(zip(vertex_list, colour_list))
        
        # Perform the colour assignment
        self.assign_vertex_colours(colour_assignment)
        
        
    # Set all edges in the graph to a given colour.
    def set_edge_colour(self, colour : str) -> None:
        
        # Get each edge and assign it the same colour - we will create a mapping here
        edge_list : list[tuple[str,str]] = list(self.E.keys())
        colour_list : list[str] = [colour] * len(edge_list)
        
        # This will be the assignment of colours (all the same colour) to assign to each edge
        colour_assignment : dict[tuple[str,str],str] = dict(zip(edge_list, colour_list))
        
        # Perform the colour assignment
        self.assign_edge_colours(colour_assignment)
        
    
    # Given a dictionary of vertex keys and colour values, for each vertex key set its colour to the corresponding colour
    def assign_vertex_colours(self, colours : dict[str,str]) -> None:
        
        # For each vertex given in the dict, we set its colour to the specified colour
        for vertex, colour in list(colours.items()):
            
            # Use the existing methods to trivialise the implementation
            self.get_vertex(vertex).set_colour(colour)
    
    
    # Given a dictionary of edge keys and colour values, for each edge set its colour to the corresponding colour    
    def assign_edge_colours(self, colours : dict[tuple[str,str] ,str]) -> None:
        
        # Same as above
        for edge, colour in list(colours.items()):
            
            # Again just use our methods to make this an easy task
            self.get_edge(*edge).set_colour(colour)
            
        
    # Remove all the highlighting from the graph
    def clear_highlighting(self) -> None:
        
        # Clear all the edges - can't get the edge because some highlighted edges don't exist in the graph
        for highlighted_edge in list(self.highlighted_edges.keys()):
            
            # By highlighting it as None we clear it
            self.highlight_edge(highlighted_edge, None)
        
        # Same idea for vertices except all vertices must exist in the graph
        for highlighted_vertex in list(self.highlighted_vertices.keys()):
            
            # By highlighting the vertices None we also clear them too
            self.highlight_vertex(highlighted_vertex, None)
        
    
    # Highlight a vertex with a given colour
    def highlight_vertex(self, vertex_name, colour : str = None,
    highlight_ratio : float = None, alpha : float = None, highlight_through : bool = None) -> None:
        
        # If these values are not set by the user, we will use the default values
        # Unfortunately we can't set predetermined values in the function definition as they are attributes
        if highlight_ratio is None: highlight_ratio = self.highlight_ratio_vertices
        if alpha is None: alpha = self.highlight_alpha
        if highlight_through is None: highlight_through = self.highlight_through
        
        # We need the properties of the vertex to 
        the_vertex : Vertex = self.get_vertex(vertex_name)
        
        # Check if we've already highlighted this vertex 
        highlighting = self.highlighted_vertices.get(vertex_name)
        
        # If there's already a highlighting we need to remove it and add a new highlighting
        if highlighting is not None:
            
            # If it does exist, remove its visual representation and then get rid of it
            highlighting.remove()
            
            # Delete it here to make sure we don't accidentally reference it again in the future
            del self.highlighted_vertices[vertex_name]

            # If they set no colour or as None then we need to remove it and then stop
            if colour is None or str(colour).lower() == "none":
                
                # Get rid of the reference and then we are done
                del the_vertex.plotrep["highlight"]
                return
        
        # Create the highlighting 
        highlight = plt.Circle((the_vertex.x, the_vertex.y), 
                               the_vertex.radius * self.highlight_ratio_vertices,
                               color = colour,
                               zorder = highlight_through * 1000,
                               alpha=alpha, clip_on = False)
        
        # Add the patch to the visual representation of the graph
        self.ax.add_patch(highlight)
        
        # Add to the list of highlighted vertices and the vertex's plot representation (so if it's removed so is the highlight)
        self.highlighted_vertices.update({ vertex_name : highlight })
        the_vertex.plotrep.update({ "highlight" : highlight })
    
    
    # Highlight an edge in the graph a given colour
    def highlight_edge(self, edge_name, colour : str = None, 
    highlight_ratio : float = None, alpha : float = None, highlight_through : bool = None) -> None:
    
        # If these values are not set by the user, we will use the default values
        # Unfortunately we can't set predetermined values in the function definition as they are attributes
        if highlight_ratio is None: highlight_ratio = self.highlight_ratio_edges
        if alpha is None: alpha = self.highlight_alpha
        if highlight_through is None: highlight_through = self.highlight_through
        
        # If the user specifies nothing or sets it to be none, then we just remove the edge and go home
        if colour == None or str(colour).lower() == "none":
            self.remove_edge_highlight(edge_name)
            return
    
        # We will look at the edge's properties to create the highlighting
        the_edge : Edge = self.get_edge(*edge_name)
        other_edge : Edge = self.get_edge(*(edge_name[::-1]))
        
        # We need to enforce particular logic if we encounter a self loop
        is_self_loop = the_edge is not None and the_edge.source == the_edge.destination
        
        # There are different cases depending on whether the edge actually exists in the graph
        if the_edge is not None:
            
            # Get the current visual representation of the edge which we will need to make highlights
            visual_edge = the_edge.plotrep["visual"]
            
            # We will handle different cases depending on the type of edge we are dealing with
            if is_self_loop:
  
                # In this case the line is actually a circle with some centre x y and radius r that we need to highlight
                highlighted_line = plt.Circle(visual_edge.center, visual_edge.radius, 
                                                alpha = alpha,
                                                linewidth = visual_edge.get_linewidth() * self.highlight_ratio_edges,
                                                edgecolor = colour,
                                                facecolor = "white",
                                                zorder = highlight_through * 1000,
                                                clip_on = False)
                
                self.ax.add_patch(highlighted_line)
                
            # If both edges (A->B, B->A) exist, it must be a bidirectional edge
            # So either a single universal edge OR 2 directional edges, either way will work
            elif other_edge is not None:
                
                # Create the highlighted line using the data of the existing line
                highlighted_line = self.ax.plot(visual_edge.get_xdata(), visual_edge.get_ydata(), 
                                            linewidth = visual_edge.get_linewidth() * self.highlight_ratio_edges,
                                            color = colour,
                                            alpha = alpha,
                                            zorder = highlight_through * 1000,
                                            clip_on = False)[0]
                
            # Otherwise it must be a single directed edge
            else:
                
                # For these arrows we need to make the multiplier 4x greater to be of the same width
                highlighted_line = self.ax.plot([the_edge.source.x, the_edge.destination.x], 
                            [the_edge.source.y, the_edge.destination.y],
                                linewidth = visual_edge.get_linewidth() * self.highlight_ratio_edges*4,
                                color = colour,
                                alpha = alpha,
                                zorder = highlight_through * 1000,
                                clip_on = False)[0]
                
            # Now that we have our highlighted lines, we can delete the original highlight if it exists
            self.remove_edge_highlight(edge_name)
            
            # Now add this highlight to the list of visual props for this edge
            the_edge.plotrep["highlight"] = highlighted_line
            
            # If it's a bidirectional edge or single directed edge then we need to add both sides of the highlight
            if not the_edge.curved and not is_self_loop:
                # We need to consider both edges to prevent colouring the same line with 2 different colours at once
                self.highlighted_edges.update({ edge_name : highlighted_line,
                                                edge_name[::-1] : highlighted_line })
                
                # If a bidirectional edge then need to also add this highlight to the other edge's list of visual props
                if other_edge is not None:
                    other_edge.plotrep["highlight"] = highlighted_line
                
            else:
                
                # If a self loop then we only need to add this case once obviously as reverse of (X,X) is (X,X)
                # Alternatively if we're curved then the other edge doesn't matter 
                self.highlighted_edges[edge_name] = highlighted_line
        
        # Otherwise if we're trying to highlight an edge that doesn't exist:    
        else: 
            
            # Get rid of any previous highlighting for this edge there may've been
            self.remove_edge_highlight(edge_name)
            
            # Highlighting self looping edges that don't exist is not allowed in this implementation
            if edge_name[0] == edge_name[1]: raise Exception("Attempted to highlight nonexistent self-loop")
            
            # We need to get the coordinates of the hypothetical source and destination vertices to create the line
            sourcev : Vertex = self.get_vertex(edge_name[0])
            destv : Vertex = self.get_vertex(edge_name[1])
            
            # Plot the highlighted straight line from the start to the end
            highlighted_line = self.ax.plot([sourcev.x, destv.x], 
                            [sourcev.y,destv.y],
                                linewidth = self.highlight_ratio_edges,
                                color = colour,
                                alpha = alpha,
                                zorder = highlight_through * 1000,
                                clip_on = False)[0]
            
            # If the edge doesn't exist then we need to add this from both directions
            self.highlighted_edges.update({ edge_name : highlighted_line,
                                edge_name[::-1] : highlighted_line })
            
            # If the other edge exists, it MUST be a single directed edge
            if other_edge is not None:
                # Add the highlighted line to its plot representation so we can reference it from there
                other_edge.plotrep["highlight"] = highlighted_line
     
       
       
    
    # Remove an edge highlighting in the graph
    def remove_edge_highlight(self, edge_name : tuple[str]) -> None:
        
        # We will look at the edge's properties to create the highlighting
        the_edge : Edge = self.get_edge(*edge_name)
        other_edge : Edge = self.get_edge(*(edge_name[::-1]))
        
        # We need to enforce particular logic if we encounter a self loop
        is_self_loop = the_edge is not None and the_edge.source == the_edge.destination
        
        # Check if there is already a highlighting present for this edge
        highlighting = self.highlighted_edges.get(edge_name)
    
        # If the highlighting doesn't exist, don't do anything
        if highlighting is not None:
            
            # First remove the visual representation
            highlighting.remove()
            
            # Then delete the reference in the highlighted edges so we don't accidentally reference it later
            del self.highlighted_edges[edge_name]
            
            # # If the reverse edge exists and isn't a self loop (otherwise we already deleted it)
            # if other_edge is not None and not is_self_loop and not the_edge.curved:
                
            # Deal with cases where the edge is and is not none separately
            if the_edge is not None:
                
                # remove it from the reference to the edge's plot representation so we don't try to delete it twice later
                del the_edge.plotrep["highlight"]
                
                if not is_self_loop and not the_edge.curved:
                    
                    # But also since any non-curved, non-selfloop edge will be referenced both ways we need to get rid of the
                    # other reference to in the highlighted edges as well
                    del self.highlighted_edges[edge_name[::-1]]
                    
                    # If we have a bidirectional edge (no self loop since we already would've removed everything before)
                    # Make sure it's not curved to prevent 2 single-directional edges from having their highlights removed
                    if other_edge is not None:
                        
                        # Remove the highlight from its references as needed
                        del other_edge.plotrep["highlight"]
                    
            # If the edge doesn't exist the logic becomes a little simpler
            else:
                
                # If the edge doesn't exist then it will be referenced both ways so delete this too
                del self.highlighted_edges[edge_name[::-1]]
                
                # If the reverse edge exists then since we've removed the highlight
                # then we need to remove its highlight representation as well
                if other_edge is not None:
                    del other_edge.plotrep["highlight"]
            
         
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
    def add_edge(self, source_name : str, destination_name : str, both : bool = False, weight : float = None,
                 edgecolour : str = None, linestyle : str = None) -> None:
        
        
        # Make the user choose an edgecolour or we use the default
        if edgecolour is None: edgecolour = self.edgecolour
        
        # Same as above but for linestyle
        if linestyle is None: linestyle = self.linestyle
        
        # Get the source and destination vertices
        sourcev : Vertex = self.get_vertex(source_name)
        destv : Vertex = self.get_vertex(destination_name)
        
        
        # We need to check to see if these edges already exist
        AtoB : Edge = self.get_edge(source_name, destination_name)
        BtoA : Edge = self.get_edge(destination_name, source_name)
        
        # If both edges already exist and we're trying to get a bidirectional, this may cause problems
        # To fix these problems, we ignore the construction and just assign the weights instead
        # This will give us identical functionality but prevent bugs
        if both and AtoB is not None and BtoA is not None:
            
            # Set both the weights to be the same
            AtoB.set_weight(weight)
            BtoA.set_weight(weight)
            
            # We are done now, we need to stop here to avoid bugs
            return
        
        # Check if we have a self loop, where the source IS the destination
        if sourcev == destv:

            # Create the selflooping arrow - we need the arrowhead
            selfloop_arrowhead, visual_edge, midpoint_x, midpoint_y = selfloop_arrow(sourcev,0.5, self.ax, edgecolour, linestyle=linestyle)
            
            # Add the edge itself
            sourcev.add_edge(sourcev, visual_edge, midpoint = [midpoint_x, midpoint_y], edgecolour=edgecolour, weight=weight,
                             linestyle=linestyle)
            
            # Grab the edge so we can add the arrowhead to it as well
            edge_just_added = self.get_edge(sourcev.name, sourcev.name)
            
            # Add the arrowhead to the edge as well so we can keep track of it
            edge_just_added.plotrep["selfloop_arrow"] = selfloop_arrowhead
            
            # Now the edge is fully added so we don't need to do anything else
            return
        
        # Add the edges using our directed edge function
        self.add_directed_edge( source_name, destination_name, weight, edgecolour=edgecolour)
        
        # Check if the other edge exists
        other_edge : Edge = self.E.get((destination_name, source_name))
        
        # If both edges exist then we remove the arrows
        if both: 
            
            # First add the other directed edge
            self.add_directed_edge( destination_name, source_name, weight, edgecolour=edgecolour)
            
            # Get both the edges
            AtoB : Edge = self.E.get((source_name, destination_name))
            BtoA : Edge  = self.E.get((destination_name, source_name))
            
            # Create the straight line
            visual_edge = self.ax.plot([sourcev.x,destv.x], [sourcev.y,destv.y], 
                                        linewidth = 1, 
                                        color=edgecolour, 
                                        zorder=0, 
                                        clip_on=False,
                                        linestyle=linestyle)[0]
            
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
            visual_arrow, visual_edge, midpoint_x, midpoint_y = curved_directed_edge_arrow(destv, sourcev, 
                                                                                           sourcev.radius * self.curved_edge_stretchiness, 
                                                                                           self.ax,
                                                                                           edgecolour=other_edge.colour,
                                                                                           linestyle=other_edge.linestyle)
            
            # Set the other edge's new midpoint
            other_edge.midpoint = [midpoint_x, midpoint_y]
            
            # The edge is now a curved directed edge so we need to update this
            other_edge.curved = True
            
            # Add its visual plot representation back
            other_edge.plotrep["visual"] = visual_edge
            other_edge.plotrep["arrow"] = visual_arrow
            other_edge.set_weight(other_edge.weight, consistent=False)
            
            # Now we will create the new edge to replace the visual representation of the edge we just added
            visual_arrow2, visual_edge2, mid2_x, mid2_y = curved_directed_edge_arrow(sourcev,destv,
                                                                                     destv.radius * self.curved_edge_stretchiness,
                                                                                     self.ax,
                                                                                     linestyle=linestyle,
                                                                                     edgecolour=edgecolour)
            
            # Now update the midpoints, visual representation and edge weight of the edge we've just added and then we're done
            edge_just_added.midpoint = [mid2_x, mid2_y]  
            edge_just_added.plotrep["arrow"] = visual_arrow2     
            edge_just_added.plotrep["visual"] = visual_edge2
            edge_just_added.set_weight(edge_just_added.weight, consistent = False)
            
            edge_just_added.curved = True

            
    # Add a directed edge to the graph - may or may not have an edge weight attached
    def add_directed_edge(self, source_name : str, destination_name : str , weight : float = None, edgecolour : str = None,
                          linestyle : str = None):
            
        # If no edgecolour is chosen let it be the default, same for linestyle
        if edgecolour is None: edgecolour = self.edgecolour    
        if linestyle is None: linestyle = self.linestyle
        
        # Get the vertices to do the edge-adding
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
        
        # If the edge already exists, ignore
        if self.E.get((source_name, destination_name)) is not None: return

        # Create an arrow axes object for use
        visual_edge = directed_edge_arrow(sourcev.x, sourcev.y, destv.x, destv.y, 
                                                    sourcev.radius, self.arrowsize, self.ax, 
                                                    edgecolour = edgecolour,
                                                    linestyle=linestyle)
            
        # Then add the directed edge to the graph structure
        sourcev.add_edge(destv, visual_edge, weight=weight, edgecolour = edgecolour, linestyle=linestyle)


    # Remove an edge from the graph; both states where it should be a bidirectional removal or just one
    def remove_edge(self, source_name : str, destination_name : str, both : bool = False) -> None:
        
        if source_name == destination_name:

            # Get the vertex and the edge so we can reference them for deletion
            the_vertex = self.get_vertex(source_name)
            the_edge = self.get_edge(source_name, source_name)
            
            if the_edge is not None:
                the_edge.plot_remove()
                
                # Delete all references to the edge so it is eligible for garbage collection
                del self.E[(source_name, source_name)]
                del the_vertex.incoming_edges[source_name]
                del the_vertex.outgoing_edges[source_name]
            
            # Now that we've fully deleted the self loop we are done, so no need to continue further
            return
        
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
            
            # If the other edge was curved (must've had a counterpart), as it now has no counterpart it cannot be curved
            other_edge.curved = False

            # Create a directed edge to account for this
            visual_edge = directed_edge_arrow(dest_vertex.x, dest_vertex.y, 
                                              source_vertex.x, source_vertex.y, 
                                              source_vertex.radius, self.arrowsize, 
                                              self.ax,
                                              edgecolour = other_edge.colour,
                                              linestyle=other_edge.linestyle)
            
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
    def add_vertex(self, vertex_name : str, x : float, y : float, radius : float, colour : str = None, 
                   textcolour : str = None) -> None:
        
        # If the user doesn't input any choice then let it be the default choice
        if colour is None: colour = self.vertexcolour
        
        # Same for the text colour of the vertex
        if textcolour is None: textcolour = self.vertex_textcolour
        
        # Instantiate the vertex
        new_vertex = Vertex(self, vertex_name, x, y, radius, colour=colour, textcolour=textcolour)
        
        # Create a circle for the vertex
        circ = plt.Circle((x,y), radius, 
                          facecolor=colour,
                          edgecolor="black",
                          zorder=1, 
                          clip_on=False )
        
        # Add the circle
        self.ax.add_patch(circ)
        
        # Link the plot representation of the vertex to the vertex itself so we can have control over it
        new_vertex.plotrep.update({ "visual" : circ })
        
        # Write the name of the vertex in the circle, centred in the circle
        vertex_text = self.ax.text(x,y, vertex_name, 
                 fontsize=150*np.pi*radius, 
                 zorder=2,
                 color=textcolour,
                 ha="center",
                 va="center")

        # Also add the text to the vertex representation as well
        new_vertex.plotrep.update({"text" : vertex_text})
        
        # Add the vertex to the graph so we can access them via their string representation
        self.V.update({ vertex_name : new_vertex })
        
        
        
        

        
        


        







