from graph_building_methods import create_graph

# GA stands for graph algorithm - it will be the algorithm handler that contains the graph we're operating on

# Determine if a graph is bipartite. If it is, then return the mappings - otherwise return False and the failed assignment
# Note that this problem is exactly equivalent to 2-colouring, so there is a trivial poly. reduction to it from this
def test_for_bipartiteness(GA, colours : list[str] = ["lime","red","grey"],
                            highlight_colour : str = "gold") -> tuple[bool, tuple[set,set,set]]:
    
    # Get the current graph being looked at by the algorithm
    G = GA.get_current_graph()
    
    # Get the original colours so we can set them back at the end
    original_colours = G.get_vertex_colours()
    
    # Start by choosing our set 1, set 2 and unassigned colours
    c1, c2, c_unassigned = colours[0], colours[1], colours[2]
    
    # Instantiate the sets which will be the partitions
    # In this mapping we have 3 keys, two for the colour partitions and one for unassigned
    # As all vertices start unassigned, we begin with all of the unassigned vertices being at c_unassigned
    partitions = { c1 : set([]), c2 : set([]), c_unassigned : GA.vertices(G)}
    
    GA.annotate(G,"Initialise all vertices as uncoloured.")
    
    # As all vertices start unassigned, give them the unassigned colour
    GA.set_vertex_colour(G,c_unassigned)
    
    # This defines what text will appear on the screen (besides annotations) explaining the state of the algorithm
    def algorithm_text() -> str:
        
        # If they are empty they just give set() which looks bad
        # Also remove the quotation marks since that's not really part of formal graph theory
        partition1 : str = str(partitions[c1]).replace("'","") if len(partitions[c1]) != 0 else "{}"
        partition2 : str = str(partitions[c2]).replace("'","") if len(partitions[c2]) != 0 else "{}"
        
        # Get rid of quotation marks and the square brackets, it needs to look like in formal graph theory
        unassigned : str = str(partitions[c_unassigned]).replace("'","").replace("[","{").replace("]","}")
        
        # This is what we will show to the user to tell them the state of the partition
        text = f"Partition 1 ({c1}): {partition1}\nPartition 2 ({c2}): {partition2}\nUnassigned: {unassigned}"
        
        return text
        
    
    # Subprocedure to assign a vertex to a partition (based on colour)
    def assign_to_partition(vertex_name, colour) -> None:
        
        # Get rid of it in unassigned
        partitions[c_unassigned].remove(vertex_name)
        
        # Add to the corresponding colour partition
        partitions[colour].add(vertex_name)
        
        # Clear the text from the previous slide
        GA.clear_text()
        
        # Explain the state of the algorithm at each step
        GA.add_text(algorithm_text())
        
        # Colour it on the graph
        GA.assign_vertex_colours(G,{ vertex_name : colour })
        
        
    # We will run a recursive DFS on the graph for bipartiteness checking
    # DFS is easier to implement and more intuitive than BFS so we use this
    def DFS_bipartite(current_vertex, current_colour) -> bool:
        
        # Assign it to the partition of the colour we've selected
        assign_to_partition(current_vertex, current_colour)
        
        # Remove the previous highlighting
        GA.clear_highlighting(G)
        
        # Get all vertices that the current vertex is connected to
        adjacent_vertices = GA.get_vertex(G,current_vertex).connected_vertices(both=True)
        
        # Now we check each vertex to see if it has a matching colour
        for vertex in adjacent_vertices:
            
            # Check if the adjacent vertex belongs to the opposite partition - if so then we break
            is_same_colour = vertex.name in partitions[current_colour]
            
            # If it's the same colour then we need to stop everything
            if is_same_colour: 
                
                # Highlight this edge with the colour to show it's not bipartite
                GA.highlight_edge(G,(current_vertex,vertex.name), current_colour)
                
                # Explain why the graph cannot be bipartite
                GA.annotate(G,f"{G.name} cannot be bipartite due to the edge {current_vertex}-{vertex.name}.")
                
                # We should save the state where it fails so the user can understand why it doesn't work
                GA.save_state()
                return False
            
        # Get the opposite colour so we can move onto the adjacent vertices and partition them
        opposite_colour = c1 if current_colour == c2 else c2
        
        # Once we are convinced there are no same-colour adjacent vertices we pick a neighbouring vertex
        for vertex in adjacent_vertices:
            
            # Check if the adjacent vertex is unassigned
            is_unassigned = vertex.name in partitions[c_unassigned]
            
            # If it's not been assigned then we recursively determine if the subgraph V \ {next vertex} is bipartite,
            # where we start off with the next vertex being coloured as the opposite colour to the current one
            if is_unassigned: 
                
                # Highlight the edge so we can see
                GA.highlight_edge(G,(current_vertex, vertex.name), highlight_colour )
                
                # Highlight the next vertex as well so the audience sees what's happening
                GA.highlight_vertex(G, vertex.name, highlight_colour)
                
                # Explain why we colour the vertex this colour.
                GA.get_vertex(G,vertex.name).annotate(f"Colour {vertex.name} {opposite_colour} because of the {current_colour} vertex {current_vertex}.")
                
                # Determine if the subgraph from exploring the next vertex is bipartite - ALL will need to be true
                subgraph_bipartite = DFS_bipartite(vertex.name, opposite_colour)
                
                # If even a single subgraph is not bipartite, it cannot be true
                if subgraph_bipartite is False: return False
                
        # If there are no unassigned adjacent vertices that are all of the opposite colour,
        # then this is a trivially true result - it must be locally bipartite
        return True
    
    # We will begin our DFS bipartiteness algorithm on some arbitrary starting vertex, the first unassigned vertex
    start_vertex = partitions[c_unassigned][0]

    # We explain to the reader what's happening
    GA.get_vertex(G,start_vertex).annotate(f"Start at the arbitrary vertex {start_vertex}.")
    
    # Highlight this vertex to show it's the start
    GA.highlight_vertex(G,start_vertex, highlight_colour)
    
    # Tells us the result of whether it's bipartite or not recursively - we always start with the first colour
    is_bipartite = DFS_bipartite( start_vertex, c1)

    # We keep doing this for every unassigned vertex - this way it works for multiple connected components
    # We only stop if we find a non-bipartite component or we have assigned all the vertices into a partition
    while is_bipartite and len(partitions[c_unassigned]) != 0:
        
        # This is the next vertex we will check for bipartiteness on
        next_start_vertex = partitions[c_unassigned][0]
        
        # If our DFS explores its connected component we need to restart at another connected component.
        GA.get_vertex(G,next_start_vertex).annotate(f"Restart algorithm at vertex {next_start_vertex}.")
        
        # Highlight this vertex to show it's the start
        GA.highlight_vertex(G,next_start_vertex, highlight_colour)
        
        # Keep performing DFS until we find every vertex
        is_bipartite = DFS_bipartite(next_start_vertex, c1)
    
    # Don't forget to give the end conclusion that it's bipartite
    if is_bipartite: 
        GA.annotate(G,f"Therefore {G.name} is bipartite, 2-colourable and has no odd-length cycles.")
        GA.save_state()
        
    # Now we build the bipartite graph to show to the user
    G2 = bipartite_form(G, partitions[c1],partitions[c2], colours[:2], name=G.name)
    
    # Switch perspective to the new graph
    GA.switch_to_graph(G2)
    
    # Explain that this is the bipartite graph and save this image
    GA.annotate(G2, f"This is the attempted bipartite construction of {G.name}.")
    GA.save_state()
    
    # Remove any remaining highlighting and return the original colours back to their existing values
    GA.clear_highlighting(G)
    G.assign_vertex_colours(original_colours)
    
    # Then at the end we return the result (yes/no), and the partitions we tried to assign to the vertices
    return (is_bipartite, (partitions[c1], partitions[c2], partitions[c_unassigned]))

# When you have to draw an edge between vertices in the same bipartite partition, make it curved
def same_partition_edges(G, source_name, dest_name, colour):
    
    # The problem is that if we draw a standard directed edge it will cut through the bipartite vertices
    # So if we have no choice but to make it curved so that it's clearly visible where the edge is going
    
    # First remove the existing edges
    G.remove_edge(source_name, dest_name, both=True)
    
    # Now add the curved edges, we add a fake edge so that we can get the curve
    G.add_edge(source_name, dest_name, edgecolour = colour)
    G.add_edge(dest_name, source_name, edgecolour = "white")
    
    # Make sure the fake edge can't overlap with anything, it will always get lower priority
    G.get_edge(dest_name, source_name).plotrep["visual"].set_zorder(-1)
    G.get_edge(dest_name, source_name).plotrep["arrow"].set_zorder(-1)


# Construct a visual representation of a bipartite graph 
# It will have all the edges of the original graph that weren't mapped to uncoloured vertices
# It will have all vertices that were coloured - belonged to one of the two partitions (L, R)
# We will use the colours list to choose our colours
def bipartite_form(G, L : set[str], R : set[str], colours : list[str] = ["lime","red"], name : str = None):
    
    # If no name is provided make it the same as G but with _bp so it's bipartite 
    if name is None: name = G.name + "_bp"
    
    # Determine the size of the partition with the most vertices, hence the tallest
    tallest = max(len(L),len(R))
    
    # This equalises their sizes with blank vertices to make implementation easier
    L_list = list(L) + (tallest-len(L))*[" "]
    R_list = list(R) + (tallest-len(R))*[" "]
    
    # The horizontal gap between the sets should always be roughly the same as the height
    # This is because the height ratio is usually 2.4, so if it's equal then height is still 2.4x width
    horizontal_gap = tallest
    
    # We will visualise the graph using the schematic to type in a string
    schematic = ""
    
    # Now we will add the vertical layout of the vertices
    # First the left vertex, the constant horizontal gap and then the right vertex
    # It will look something like this:
    
    # A         H
    # B         I
    # C         J
    # D         K
    # E         L
    # F
    # G
    
    for i in range(tallest):
        # Implements the above visual representation for the schematic
        schematic += L_list[i] + horizontal_gap * " " + R_list[i]
        
        # Don't forget to add newlines(unless we're at the end)
        if i != tallest-1: schematic += "\n"
    
    # Create the graph using no edges or weights, since we will add these now 
    # Give the graph the same name so it gets put in the same folder 
    G2 = create_graph(schematic, "", [], name=G.name)
    
    # Create colour mappings for the left and right partitions respectively
    # So every vertex in a partition will get that partition's colour
    colour_assignment_L = dict(zip( list(L) , [colours[0]] * len(L) )) 
    colour_assignment_R = dict(zip( list(R), [colours[1]] * len(R) ))

    # Assign the mappings to the graph
    G2.assign_vertex_colours(colour_assignment_L)
    G2.assign_vertex_colours(colour_assignment_R)
    
    for source_name, dest_name in G.edges():
        
        # Check if the original edge was bidirectional or not
        original_edge = G.get_edge(source_name, dest_name)
        
        # Check if it was a bidirectional edge - hence both edges must exist and they cannot be curved
        is_bidirectional = not original_edge.curved and G.get_edge(dest_name, source_name) is not None
        
        # First we check that both the source and destination are part of either the left or the right partitions
        if (set([source_name, dest_name]).issubset(L.union(R))  ):
            
            # If they are then we can safely add the edge
            
            # Add the edge using the information provided
            G2.add_edge(source_name, dest_name, is_bidirectional)
        
            # If they are from L to L or R to R then we colour the edges the opposite colour to show it's not bipartite
            if source_name in L and dest_name in L:
                same_partition_edges(G2, source_name, dest_name, colours[1])
            elif source_name in R and dest_name in R:
                same_partition_edges(G2, source_name, dest_name, colours[0])
    
    # Now we have finished building the graph so we can give it back to the user
    return G2