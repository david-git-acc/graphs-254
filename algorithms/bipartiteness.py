# GA stands for graph algorithm - it will be the algorithm handler that contains the graph we're operating on

# Determine if a graph is bipartite. If it is, then return the mappings - otherwise return False and the failed assignment
# Note that this problem is exactly equivalent to 2-colouring, so there is a trivial poly. reduction to it from this
def test_for_bipartiteness(GA, colours : list[str] = ["lime","red","grey"],
                            highlight_colour : str = "gold") -> tuple[bool, tuple[set,set,set]]:
    
    # Get the current graph being looked at by the algorithm
    G = GA.get_current_graph()
    
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
        partition1 : str = str(partitions[c1]) if len(partitions[c1]) != 0 else "{}"
        partition2 : str = str(partitions[c2]) if len(partitions[c2]) != 0 else "{}"
        
        # This is what we will show to the user to tell them the state of the partition
        text = f"Partition 1 ({c1}): {partition1}\nPartition 2 ({c2}): {partition2}\nUnassigned: {str(partitions[c_unassigned])}"
        
        return text
        
    
    # Subprocedure to assign a vertex to a partition (based on colour)
    def assign_to_partition(vertex_name, colour) -> None:
        
        # Add to the corresponding colour partition
        partitions[colour].add(vertex_name)
        
        # Colour it on the graph
        GA.assign_vertex_colours(G,{ vertex_name : colour })
        
        # Get rid of it in unassigned
        partitions[c_unassigned].remove(vertex_name)
        
    # We will run a recursive DFS on the graph for bipartiteness checking
    # DFS is easier to implement and more intuitive than BFS so we use this
    def DFS_bipartite(current_vertex, current_colour) -> bool:
            
        GA.clear_text()
        
        # Explain the state of the algorithm at each step
        GA.add_text(algorithm_text())
            
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
    
    # Then at the end we return the result (yes/no), and the partitions we tried to assign to the vertices
    return (is_bipartite, (partitions[c1], partitions[c2], partitions[c_unassigned]))