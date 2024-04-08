from graph import Graph
from algorithms.dfs import dfs
from helper_functions import get_n_colours

def construct_metagraph(G : Graph, partitions : list[list[str]],
         names : list[str] = None, colours : list[str] = None,
         metagraph_name : str = None) -> Graph:
    
    """
    Given a graph and a list of disjoint vertex partitions, construct the "metagraph" by
    merging (contracting) the vertices in each partition into a single vertex. 
    
    This will not impact the original graph - the work will be performed on a copy
    and provided as the return value of this function.
    
    Args:
        G (Graph): The graph to compute the metagraph of.
        partitions (list[list[str]]): the list of vertex partitions.
        Each partition is a list of vertex names (strings).
        names (list[str]): the names of the metagraph/contracted vertices.
        colours (list[str]): the colours of the metagraph/contracted vertices.
        metagraph_name (str): the name of the new metagraph. Defaults to (graph_name)_meta.
        
    Raises:
        Exception: if the vertex partitions given are not disjoint from each other

    Returns:
        Graph: The metagraph formed 
    """
    
    if metagraph_name is None: metagraph_name = G.name + "_meta"
    
    # Checking for partition disjointness
    concat = [vertex for partition in partitions for vertex in partition]
    if len(set(concat)) != len(concat): raise Exception(f"Partitions of {G.name} are not disjoint")
    
    # Create a copy of the graph so we don't affect the original
    G_meta = G.clone(metagraph_name)
    
    # Get the number of partitions for reference
    n = len(partitions)
    
    # If no SCC names are given, just write them by number, starting from 1
    if names is None: names = [f"S{i+1}" for i in range(n)]
    
    # If no colours given then we can just get them normally
    if colours is None: colours = get_n_colours(n)
    
    # The metagraph is the contraction of the vertices
    for partition_index, partition in enumerate(partitions):
        contract_vertices(G_meta, partition, names[partition_index], 
                          vertex_colour= colours[partition_index])
        
    return G_meta

    


def contract_vertices(G : Graph, vertex_list : list[str], contracted_vertex_name : str, 
                      vertex_radius : float = None, vertex_colour : str = None,
                      vertex_textcolour : str = None) -> None:
    
    """
    Given a list of vertices, combine them into a single vertex such that the new vertex
    shares all the edges of the original vertices and their attributes. 

    Args:
        G (Graph): The graph to combine the vertices in.
        vertex_list (list[str]): The list of vertex names to contract.
        contracted_vertex_name (str): The name of the new contracted vertex.
        vertex_radius (float, optional): The size (radius) of the contracted vertex. Defaults to None.
        vertex_colour (str, optional): The colour of the new vertex. Defaults to None.
        vertex_textcolour (str, optional): The colour of the vertex's text colour. Defaults to None.
    """
    
    # If we try to contract 0 vertices then it's equivalent to doing nothing    
    if len(vertex_list) == 0: return
    
    # If there's only 1 vertex then we just rename it and and we're done
    if len(vertex_list) == 1: 
        G.get_vertex(vertex_list[0]).rename(contracted_vertex_name)
        return
    
    # These coords will determine the coordinates of the contracted vertex (mean)
    avg_xcoords = avg_ycoords = 0
    
    # We store the outgoing and incoming vertices of the new contracted vertex in a set
    # This way we avoid having parallel edges, where we have 2 edges in the contracted vertex with 
    # the same start and end points, which is not permitted in this graph program
    incoming_vertices = set([])
    outgoing_vertices = set([])

    # Get it in set format for O(1) membership
    vertex_set = set(vertex_list)
    
    # Determine the colour of the contracted vertex
    if vertex_colour is None: vertex_colour = G.vertexcolour
    if vertex_textcolour is None: vertex_textcolour = G.vertex_textcolour
    
    for vertex_name in vertex_list:
        
        vertex = G.get_vertex(vertex_name)
        
        # Also get the radius of the vertex to make for our new contracted vertex
        if not vertex_radius: vertex_radius = vertex.radius
        
        # The new vertex's coordinates will be the mean x and y coordinates of all its component vertices
        avg_xcoords += vertex.x
        avg_ycoords += vertex.y
        
    # Now we have the coordinates of the new contracted vertex        
    avg_xcoords /= len(vertex_list)
    avg_ycoords /= len(vertex_list)
    
    # Adding the contracted vertex to the graph
    G.add_vertex(contracted_vertex_name, avg_xcoords, avg_ycoords, vertex_radius, vertex_colour, G.vertex_textcolour )
    
    # We have to add another FOR, because we can only clone edges for vertices that exist in the graph
    # Before, we didn't know the x and y coords of the contracted vertex so we couldn't copy the edges over    
    for vertex_name in vertex_list:
        
        vertex = G.get_vertex(vertex_name)
    
        # The new vertex must have all the outgoing and incoming edges as all of its component vertices   
        # IGNORE any internal edges - that is, edges between vertices in the contracted set     
        for outgoing_vertex in list(vertex.outgoing_edges.keys()):
            
            # Internal edges will disappear, and we need to avoid parallel edges so check it's not already there
            if outgoing_vertex not in vertex_set and outgoing_vertex not in outgoing_vertices:
                
                # Copy the edge over so that the new contracted vertex inherits the edge 
                G.clone_edge((vertex_name, outgoing_vertex), (contracted_vertex_name, outgoing_vertex))
                
                # Add it to the set so we don't add it again
                outgoing_vertices.add(outgoing_vertex)
    
        # This is the same principle as above, just for incoming edges
        for incoming_vertex in list(vertex.incoming_edges.keys()):
            
            # Check it's not already been added to avoid parallel edges of course
            if incoming_vertex not in vertex_set and incoming_vertex not in incoming_vertices:
                
                G.clone_edge((incoming_vertex, vertex_name),(incoming_vertex, contracted_vertex_name))
                
                # Add it to the set so we don't add it again
                incoming_vertices.add(incoming_vertex)
                
        # Now that we've copied all the information over, we don't need this vertex anymore so we delete it
        G.remove_vertex(vertex_name)
        
        

def reverse_graph(G : Graph, graph_name : str = None, preserve_highlights : bool = True) -> Graph:
    """
    Given a graph G, reverse all of its edges and return the reversed graph.
    No effect on bidirectional or self-loop edges. The original graph is not modified.

    Args:
        G (Graph): The graph whose edges are to be reversed
        graph_name (str, optional): The new name of the reversed graph. Defaults to None.
        preserve_highlights (bool, optional): whether to keep the highlighting of the original graph. Defaults to True.

    Returns:
        Graph: A copy of G whose edges travel opposite to their original directions.
    """
    
    # Default name is just name of the graph + reverse
    if graph_name is None: graph_name = G.name + "_rev"
    
    # Clone the graph so we can make modifications to it without affecting the original graph
    G_rev : Graph = G.clone( graph_name, preserve_highlights=False)

    # The highlights of the reverse graph will be the symmetric opposites since the graph will be the reverse
    G_edge_highlights = { edge_name[::-1] : highlight_info for edge_name, highlight_info 
                         in list(G.get_all_edge_highlight_properties().items()) }
    
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



def scc_algorithm(GA, start_vertex_name : str = None, highlight_colour : str = "gold", 
        finish_colours : list[str] = ["red","lime"]) -> tuple[list[list[str]], Graph]:
    """
    Algorithm to compute the Strongly Connected Components (SCCs) of a graph.

    Args:
        GA (_t): _description_
        start_vertex_name (str, optional): The vertex to begin the first DFS search from. If no name is given, one will be selected arbitrarily.
        highlight_colour (str, optional): The colour to highlight the DFS forest with. Defaults to "gold".
        finish_colours (list[str], optional): The colours to highlight the unexplored and finished vertices in DFS with. Defaults to ["red","lime"].

    Returns:
        list[list[str]]: The list of partitions, themselves represented as lists. Each vertex in a partition is given by its string name. 
        Graph: the SCC metagraph produced by the SCCs of G. This does not modify the original graph.
    """
    
    # Get the current graph in use by the graph algorithm to find the SCCs for
    G = GA.get_current_graph()
    
    # Store the original colours so we can add them back after we're done
    original_vertex_colours = G.get_vertex_colours()
    original_edge_colours = G.get_edge_colours()
    
    # Remove all existing highlights to avoid confusion
    G.clear_highlighting()
    
    # This dictionary will store as the key the partition name, and as the value the set of vertices in the partition
    partitions = {}
    
    # Also store the explored vertices so we don't accidentally explore the same vertices multiple times
    explored = []
    
    # Store all coloured edges so any edges left uncoloured will be known to be metagraph edges
    coloured_edges = set([])
    
    # For visualisation we need to know the number of SCCs ahead of time so we know how many colours to use
    # Default to the number of vertices as there can never be more SCCs than vertices
    num_sccs = len(G.V)
    if GA.capturing:
        # Determine the number of SCCs ahead of time by running the algorithm
        num_sccs = len(GA.run_algorithm(scc_algorithm, graph=G, capture=False)[0])
        
        # This will be set to false by the run of the algorithm so we need to set it back to True again
        GA.capturing=True
    
    # Obtain the colours for the SCCs for visualisation purposes
    scc_colours = get_n_colours(num_sccs)
    
    # This is the text that we will show at each step of our algorithm
    def algorithm_text(finish_numbers) -> str:
        
        # These are the finish numbers that are still remaining that we need to explore with DFS
        textstring = "Remaining vertices and finish numbers: " + str(finish_numbers).replace("'","").replace("[","{").replace("]","}") + "\n"
        
        if len(partitions) != 0:
        
            for partition_name, partition_set in list(partitions.items()):
                setstring = str(partition_set).replace("'","") if len(partition_set) != 0 else "{}"
                textstring += partition_name + " : " + setstring + "   "
        
            return textstring[:-1]
        
        else: return textstring + "No partitions created yet."
    
    if start_vertex_name is None: start_vertex_name = G.vertices()[0]
    
    # No vertices or finish numbers found yet
    GA.annotate(G, f"To identify the SCCs of the graph, we will first need to create the reverse graph {G.name}'.")
    GA.save_state()
    
    # Create the reverse graph for us to use Kosaraju's algorithm on
    G_rev = reverse_graph(G, G.name, preserve_highlights=False)
    GA.switch_to_graph(G_rev)
    
    GA.annotate(G_rev , f"This is the reverse graph {G.name}' - begin with a depth-first-search from vertex {start_vertex_name}.")
    GA.save_state()
    GA.save_state() # Save twice so user has more time to see it
    
    # Run DFS to get the DFS finish numbers
    DFS_finish_numbers, _,_ = GA.run_algorithm(dfs, start_vertex_name, graph=G_rev, highlight_colour=highlight_colour, 
                                          finish_colours=finish_colours, kill_existing=False, capture=GA.capturing, save_video = False )
    
    # Sort the finish numbers by descending order of finish numbers so we know which vertices to start at
    sorted_finish_numbers = sorted(list(DFS_finish_numbers.items()), key = lambda x : x[1], reverse=True)
    
    # Now we will switch back to the original graph G to start searching for SCCs
    GA.switch_to_graph(G)
    GA.clear_text()
    GA.add_text(algorithm_text(sorted_finish_numbers))
    GA.annotate(G,f"Now we will run DFS on {G.name} in descending order of vertex finish numbers to find SCCs.")
    GA.save_state()
    
    # Count the number of partitions with this counter
    i=0
    
    while len(sorted_finish_numbers) != 0:
        
        # Get the name of the next vertex to run DFS from - we go by descending order of finish numbers
        # As the finish numbers have been sorted, this means it'll always be the first one
        next_start_vertex = sorted_finish_numbers[0][0]
        
        GA.clear_text()
        GA.add_text(algorithm_text(sorted_finish_numbers))
        GA.annotate(G, f"Vertex {next_start_vertex} has the maximal finish number, so begin DFS from there to find an SCC.")
        GA.save_state()
        
        # Now run DFS again on this vertex to identify SCCs - we don't care about the finish numbers anymore
        _, next_SCC_vertices,_ = GA.run_algorithm(dfs, start_vertex_name=next_start_vertex, graph=G, persistent=False, highlight_colour=
                                                highlight_colour, finish_colours=finish_colours, kill_existing=False,
                                                pre_explored = explored, capture=GA.capturing, save_video = False,
                                                skip_intro = True)
        
        # Add these vertices to the list of explored vertices so we don't explore them again
        explored += next_SCC_vertices
        
        # Get the colour for this SCC for the visualisation
        this_SCC_colour = scc_colours[i]
        
        # Increment i as we've found another SCC
        i+=1
        
        # Convert into a set - this will be useful for edge colouring
        SCC_vertex_set = set(next_SCC_vertices)
        
        # Add this partition to the set of SCCs 
        partitions.update({ f"SCC {i}" : SCC_vertex_set }) 
        
        # Now we will remove every vertex explored in this SCC from the set of vertices to be explored
        # So that we don't accidentally explore the same vertex more than once and so we identify different SCCs
        for SCC_vertex in next_SCC_vertices:
            
            # Get the finish number so we can get the tuple pair of (vertex, finish number)
            # By getting this tuple pair we can then delete it from the list of sorted vertices 
            finish_number = DFS_finish_numbers.get(SCC_vertex)
            
            # Remove from consideration so we can explore the next vertices
            sorted_finish_numbers.remove((SCC_vertex, finish_number))
            
            # Now we want to colour all edges whose endpoints are both in this SCC the same colour as this SCC
            adjacent_vertices = G.get_vertex(SCC_vertex).connected_vertices()
            
            for adj_vertex in adjacent_vertices:
                
                if adj_vertex.name in SCC_vertex_set:
                    G.get_edge(SCC_vertex, adj_vertex.name).set_colour( this_SCC_colour)
                    coloured_edges.add((SCC_vertex, adj_vertex.name))
        
        # Each SCC will be given a different colour, so that we can visually distinguish them
        colour_assignment = dict(zip(next_SCC_vertices, [this_SCC_colour] * len(next_SCC_vertices)))
        
        GA.add_text(algorithm_text(sorted_finish_numbers))
        GA.annotate(G, f"SCC {i} has been found - colour it accordingly.")
        
        # Perform the colour assignment
        GA.assign_vertex_colours(G, colour_assignment)
        
    # These are all the edges which will be used in the metagraph
    metagraph_edges = list(set(G.edges()) - coloured_edges)    
    
    # Any edge of the graph that isn't already coloured must be a metagraph edge
    for edge_name in metagraph_edges:
        # We will make the highlight the same colour as the source colour so we can see where the SCCs leave
        
        source_colour = G.get_vertex(edge_name[0]).colour
        G.highlight_edge(edge_name, source_colour, alpha = 0.25)
        
    GA.annotate(G, f"We have now found all {i} connected components of {G.name}.")
    GA.save_state()
    
    # Get the partitions into the form we want before we give them back
    partition_form = [ list(partition) for partition in list(partitions.values())]
    
    # Build the SCC metagraph of G
    G_meta = construct_metagraph(G, partition_form, None, scc_colours, metagraph_name=G.name)
    
    # Switch to the metagraph so we can see it
    GA.switch_to_graph(G_meta)
    
    GA.annotate(G_meta, f"This is the SCC metagraph of {G.name}.")
    GA.add_text(algorithm_text(sorted_finish_numbers))
    GA.save_state()
    
    # Now go back to the original graph
    GA.switch_to_graph(G)
    
    # Restore the original vertex colours for the graph
    G.assign_vertex_colours(original_vertex_colours)
    G.assign_edge_colours(original_edge_colours)
    
    # Get rid of this text so the next algorithm can start
    GA.clear_highlighting(G)
    GA.clear_annotations(G)
    GA.clear_text()
    
    # Rename the graph so it's not confused with the original graph
    G_meta.rename(G.name + "_meta")
    
    return (partition_form,G_meta)