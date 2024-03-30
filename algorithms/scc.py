from graph import Graph

# Given a graph G, reverse all of its edges and return the reversed graph
def reverse_graph(G : Graph, graph_name : str = None, preserve_highlights : bool = True) -> Graph:
    
    # Default name is just name of the graph + reverse
    if graph_name is None: graph_name = G.name + "_rev"
    
    # Clone the graph so we can make modifications to it without affecting the original graph
    G_rev : Graph = G.clone( graph_name, preserve_highlights=False)

    # The highlights of the reverse graph will be the symmetric opposites since the graph will be the reverse
    G_edge_highlights = { edge_name[::-1] : highlight_info for edge_name, highlight_info in list(G.get_all_edge_highlight_properties().items()) }
    
    # The vertices are not affected by this transformation, so the highlight properties are the same
    G_vertex_highlights = G.get_all_vertex_highlight_properties()
    
    # This set stores the names of all edges which have already been reversed to avoid repetitions
    already_reversed = set([])
    
    # Now we will go through each edge and reverse them accordingly
    for edge in G.edges(objects=True):
        
        # Get the name of the edge so we can refer to it later
        edge_name = (edge.source.name, edge.destination.name)
        
        # For self loops and bidirectional edges, we don't have to do anything to reverse them
        if not edge.is_self_loop() and not edge.is_bidirectional() and not edge_name in already_reversed:
            
            # Add to the set of reversed edges so we don't reverse them again
            already_reversed.add(edge_name)
            
            # Store all the properties of this edge so we can add them again
            this_edge_props = [edge.weight, edge.colour, edge.textcolour, edge.linestyle ]
            
            # Now we will check to see if the other edge exists
            other_edge_name = edge_name[::-1]
            
            # Ge the other edge
            other_edge = G.get_edge(*other_edge_name)
            
            if other_edge is not None:
            
                # If the other edge exists then since we're handling it now we won't handle it again later, so add it to
                # the list of already reversed edges in the graph
                already_reversed.add(other_edge_name)
                
                # Get all the properties of the other edge as well for the same reason
                other_edge_props = [other_edge.weight, other_edge.colour,other_edge.textcolour, other_edge.linestyle]

                # Remove the edge from the graph, so we can add the edges back in reverse order
                G_rev.remove_edge(*edge_name, both=True)

                # Now we will add the edge from source to destination having the properties of the edge from dest to source
                G_rev.add_edge(*edge_name, both=False,
                               weight=other_edge_props[0], edgecolour = other_edge_props[1], 
                               linestyle=other_edge_props[3])
                
                # We have to update the textcolour separately
                G_rev.get_edge(*edge_name).set_textcolour(other_edge_props[2])
                
                # And vice versa - by giving the other edge's properties to the new added edge, we are effectively
                # reversing the edges
                G_rev.add_edge(*other_edge_name, both=False,
                               weight = this_edge_props[0], edgecolour=this_edge_props[1],
                               linestyle = this_edge_props[3])
                
                # And again set the textcolour separately
                G_rev.get_edge(*other_edge_name).set_textcolour(this_edge_props[2])
                
            # Otherwise it's just a straight directed edge A-->B with no counterpart B-->A, so we reverse this edge
            else:
                
                # Remove the edge and we will add it in reverse
                G_rev.remove_edge(edge.source.name, edge.destination.name)
                
                # Now add the edge back on pointing in the opposite direction
                G_rev.add_edge(edge.destination.name, edge.source.name, both=False,
                               weight = this_edge_props[0], edgecolour=this_edge_props[1],
                               linestyle = this_edge_props[3])
                
                # And again set the textcolour separately
                G_rev.get_edge(*other_edge_name).set_textcolour(this_edge_props[2])
                
        else:
            # If bidirectional/self-loop then just add it to already reversed without making any changes
            already_reversed.add(edge_name)
    
    # Add G's original highlights to the reverse graph - reversed edges will keep their original highlightingss            
    if preserve_highlights:
        G_rev.assign_vertex_highlights(G_vertex_highlights)
        G_rev.assign_edge_highlights(G_edge_highlights)
            
    return G_rev


def scc_algorithm(GA, start_vertex_name = None, highlight_colour : str = "gold", 
        finish_colours : list[str] = ["red","lime"]) -> list[list[str]]:
    
    # Get the current graph in use by the graph algorithm to find the SCCs for
    G = GA.get_current_graph()
    
    GA.annotate(G, f"To identify the SCCs of the graph, we will first ")
