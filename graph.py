import numpy as np
import matplotlib.pyplot as plt
from drawing_functions import directed_edge_arrow, curved_directed_edge_arrow, selfloop_arrow

# Defining our graph theory objects
class Graph():
    def __init__(self, name, ax, arrowsize : float =0.01, res : int =100, vertexcolour : str ="red",edgecolour : str="black",
                 textcolour : str = "black", curved_edge_stretchiness : float = 1.4):
        self.name = name
        
        # V maps the NAME of the vertex to the object vertex
        # E.g for a vertex called V, then we have an entry { "V" : (Object Vertex() at 0x3299239230) }
        self.V : dict[str, Vertex] = {}
        
        # E maps a pair (source name, destination name) to the actual object edge
        # E.g for vertices called U and V, then we have an entry { ("U", "V") : (Object Edge() at 0x2392230230)} 
        self.E : dict[tuple[str,str], Vertex] = {}
        
        # Get a reference to the axis
        self.ax = ax
        
        # MISCELLANEOUS ATTRIBUTES - used in graph customisation
        
        # Determine the default colour of our vertices and edges
        self.vertexcolour = vertexcolour
        self.edgecolour = edgecolour
        
        # The textcolour for all the weights and vertices
        self.textcolour = textcolour
        
        # Set the arrowsize - the size of the arrows for directed edges
        self.arrowsize = arrowsize
        
        # This constant determines how much curved edges will be stretched out
        self.curved_edge_stretchiness = curved_edge_stretchiness
        
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
    def add_edge(self, source_name : str, destination_name : str, both : bool = False, weight : float = None,
                 edgecolour : str = None) -> None:
        
        # Make the user choose an edgecolour or we use the default
        if edgecolour is None: edgecolour = self.edgecolour
        
        # Get the source and destination vertices
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
        
        # Check if we have a self loop, where the source IS the destination
        if sourcev == destv:
            print("Selfloop!" , sourcev.name)
            
            # Create the selflooping arrow - we need the arrowhead
            selfloop_arrowhead, visual_edge, midpoint_x, midpoint_y = selfloop_arrow(sourcev,0.5, self.ax, edgecolour)
            
            # Add the edge itself
            sourcev.add_edge(sourcev, visual_edge, midpoint = [midpoint_x, midpoint_y], edgecolour=edgecolour, weight=weight)
            
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
        
        # # If we want 2 directed edges but they have the same weight then there's no point in making a curved line
        # # So we will make them the same edge instead so that the graph is not unnecessarily complicated
        # if not both and weight is not None and other_edge is not None and other_edge.weight == weight :
        #     both = True
            

        

            
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
            visual_arrow, visual_edge, midpoint_x, midpoint_y = curved_directed_edge_arrow(destv, sourcev, 
                                                                                           sourcev.radius * self.curved_edge_stretchiness, 
                                                                                           self.ax,
                                                                                           edgecolour=other_edge.colour)
            
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
                                                                                     edgecolour=edgecolour)
            
            # Now update the midpoints, visual representation and edge weight of the edge we've just added and then we're done
            edge_just_added.midpoint = [mid2_x, mid2_y]  
            edge_just_added.plotrep["arrow"] = visual_arrow2     
            edge_just_added.plotrep["visual"] = visual_edge2
            edge_just_added.set_weight(edge_just_added.weight, consistent = False)
            
            edge_just_added.curved = True

            
    # Add a directed edge to the graph - may or may not have an edge weight attached
    def add_directed_edge(self, source_name : str, destination_name : str , weight : float = None, edgecolour : str = None):
            
        # If no edgecolour is chosen let it be the default
        if edgecolour is None: edgecolour = self.edgecolour    
        
        # Get the vertices to do the edge-adding
        sourcev : Vertex = self.V.get(source_name)
        destv : Vertex = self.V.get(destination_name)
        
        # If the edge already exists, ignore
        if self.E.get((source_name, destination_name)) is not None: return

        # Create an arrow axes object for use
        visual_edge = directed_edge_arrow(sourcev.x, sourcev.y, destv.x, destv.y, 
                                                    sourcev.radius, self.arrowsize, self.ax, 
                                                    edgecolour = edgecolour)
            
        # Then add the directed edge to the graph structure
        sourcev.add_edge(destv, visual_edge, weight=weight, edgecolour = edgecolour)


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
                                              edgecolour = other_edge.colour)
            
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
    def add_vertex(self, vertex_name : str, x : float, y : float, radius : float, colour : str = None) -> None:
        
        # If the user doesn't input any choice then let it be the default choice
        if colour is None: colour = self.vertexcolour
        
        # Instantiate the vertex
        new_vertex = Vertex(self, vertex_name, x, y, radius, colour=colour)
        
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
    def __init__(self, G : Graph, name : str, x : float, y : float, radius : float, colour : str = None):
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
        
        # The colour of the vertex - can be overridden by the user if necessary
        if colour is None:
            self.colour = G.vertexcolour
        else:
            # If the user inputs a colour choice then respect that choice
            self.colour = colour
            
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
    def add_edge(self, dest, visual_edge, weight : float = None, midpoint : list[float,float] = None, 
                 edgecolour : str = None):
        
        # If no edgecolour selected, let it be the default
        if edgecolour is None: edgecolour = self.owner.edgecolour
        
        # Make sure this edge actually exists
        if not isinstance(dest, Vertex):
            raise Exception(f"Destination vertex for outgoing edge from {self.name} does not exist")
        
        # Instantiate the edge with all the information
        edge = Edge(self, dest, weight, midpoint=midpoint, colour=edgecolour)
        
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
        
        

# Defining an edge
# Every vertex has a source and destination vertex, and a midpoint (which may be specified manually if the user wants)
# Also we have an optional weight which can be used
# The curved attribute determines whether this edge is a curved directed edge or not
class Edge():
    def __init__(self, source : Vertex, destination : Vertex, weight : float = None, midpoint : list[float,float] = None,
                 curved : bool = False, colour : str = None):
        
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
    def plot_remove(self):
        
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
        







