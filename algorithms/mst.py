from graph import Graph
from collections import deque
from numpy import number


def red_rule(graph : Graph,  start_vertex_name : str = None,  red_colour : str = "red",
             colour_edge : bool = True) -> list[tuple[str,str], list[tuple[str,str]]]:
    
    """
    Perform the red rule on an undirected, connected, weighted graph. 
    The red rule is defined as identifying a cycle with no red-coloured edges and colouring red the edge in
    the cycle of maximal weight.
    
    A double breadth-first-search is used to identify the closest proximity red-edgeless cycle to the starting vertex.
    
    Args:
        graph (Graph): the graph to perform the red rule on.
        start_vertex_name (str): The graph to start the search on. This influences which cycle is chosen. Defaults to some arbitrary vertex.
        red_colour (str) : the choice of colour to use to identify edges as "red". Defaults to red.
        colour_edge (bool) : whether to colour the selected edge red in the graph or not. Defaults to True.

    Returns: a tuple, where the first element is the edge of maximal weight (coloured red) and the second is the list of edges 
    in the red cycle. If no such red-cycle can be found, returns None for both elements.
    """
    
    # If no start vertex specified, select one arbitrarily
    if start_vertex_name is None: start_vertex_name = graph.vertices()[0]
    
    # This is a dictionary where the key is a vertex and the value is all connected vertices in the BFS tree
    bfs_edges : dict[str,list[str]] = { vertex_name : [] for vertex_name in graph.vertices() }

    # Set of explored vertices to help us identify cycles
    explored = set([start_vertex_name])
    
    # BFS visiting queue
    visiting = deque([start_vertex_name])
    
    # Store the parent of each vertex - that way we don't go back on ourselves
    # and try to say there's a cycle where we're actually just going in reverse
    parent_vertices = { start_vertex_name : start_vertex_name }
    
    # We terminate the algorithm as soon as we find a cycle    
    cycle_found = False

    # BFS is the best algorithm for finding the smallest possible cycles
    while len(visiting) != 0 and not cycle_found:
        
        # Get the next vertex to explore in the graph
        current_vertex_name = visiting.popleft()
        current_vertex = graph.get_vertex(current_vertex_name)
        
        # We explore each neighbour in an attempt to find a cycle not containing a red edge
        neighbours = current_vertex.connected_vertices()
        
        for neighbour in neighbours:
            
            # Don't try to form a cycle with the parent of this vertex or else this creates a 2-cycle
            # which is not allowed as all undirected cycles are of length >2
            if parent_vertices.get(current_vertex_name) != neighbour.name:
            
                # Get the edge connecting these vertices - we should not include any red edges in our cycle
                vertex_neighbour_edge = graph.get_edge(current_vertex_name, neighbour.name)
                
                # The red rule states that our cycle cannot contain red edges
                if vertex_neighbour_edge.colour != red_colour:
                    
                    # Add to the list of BFS edges so we can use it to check for cycles later
                    bfs_edges[current_vertex_name].append(neighbour.name)
                    bfs_edges[neighbour.name].append(current_vertex_name)
                    
                    # Update the parent mapping so we avoid backtracking to the parent
                    parent_vertices[neighbour.name] = current_vertex_name
                    
                    # Add to visiting so we can explore this path as well
                    visiting.append(neighbour.name)
                    
                    # If we find a vertex we've already found before then we've found a simple cycle, so we can stop here
                    if neighbour.name in explored:
                        # We let this neighbour be the start point to explore the cycle from
                        cycle_found = True
                        break
                    
                    # Otherwise add to set of explored vertices and keep going
                    else: explored.add(neighbour.name)
   
    
        # If cycle found, we remove all "tree edges" - that is, edges not part of the cycle
        # As our BFS edges will be a superset of the cycle as we will have travelled along some other edges along the way
        if cycle_found:
            
            # First get all vertices of degree one - we will remove them one by one until none left
            degree_one_vertices = [ vertex_name for vertex_name, conn_vertices in list(bfs_edges.items()) if len(conn_vertices) == 1]
            
            # We will now remove all edges where one of the vertices has degree 1 - such edges can never be part of the cycle
            for vertex_name in degree_one_vertices:
                
                # As it's a degree one vertex it has only one vertex connection - this is the connected vertex 
                connected_vertex = bfs_edges[vertex_name][0]
   
                # Now remove the edge from the cycle - degree one edges may never be part of the cycle
                bfs_edges[vertex_name].remove(connected_vertex)
                bfs_edges[connected_vertex].remove(vertex_name)
                
                # We keep repeating this process until we have removed all edges where one of the endpoints has degree 1
                # Of course deleting edges in this way will create new vertices of degree 1, so we have to keep doing this
                while len(bfs_edges[connected_vertex]) == 1:
                    
                    # Get the vertex the current degree one vertex is connected to
                    new_connected_vertex = bfs_edges[connected_vertex][0]
                    
                    # Remove the edge - as it's bidirectional we have to remove it from both cases
                    bfs_edges[new_connected_vertex].remove(connected_vertex)
                    bfs_edges[connected_vertex].remove(new_connected_vertex)
                    
                    # Repeat this process again and again until we find a vertex with multiple connections
                    connected_vertex = new_connected_vertex
                    
                    
            # Now all the remaining BFS edges must form part of the cycle - we store these edges here
            cycle_edges = []                    
            
            # For each edge remaining it must form part of the cycle so we add it to cycle edges
            # As it's in adjacency list format we add it like this
            for vertex_name, conn_vertices in list(bfs_edges.items()):
                for conn_vertex in conn_vertices:
                    cycle_edges.append((vertex_name, conn_vertex))
                    
            # Get each edge name and its corresponding weight
            edge_names_and_weights = [(edge_name, graph.get_edge(*edge_name).weight) for edge_name in cycle_edges]
                
            # Obtain the maximal weight edge which we will colour red
            maximal_weight_edge, _ = max(edge_names_and_weights, key = lambda x : x[1])    
            
            # Now colour the maximal weight edge red
            if colour_edge: graph.get_edge(*maximal_weight_edge).set_colour(red_colour)
                
            # In this case, a cycle has indeed been found, so we return True and the maximal weight edge 
            return ( maximal_weight_edge, cycle_edges)
    
    # Otherwise no red-rule cycle exists so return None for both
    return (None, None)
            
            
   
            
            
def blue_rule(graph : Graph, start_vertex_name : str = None, 
              blue_colour : str = "blue", colour_edge : bool = True) -> tuple[tuple[str,str], list[tuple[str,str]]]:

    """
    Perform the blue rule on an undirected, connected, weighted graph.
    The blue rule is defined as dividing the vertices of the graph into two partitions such that no blue-coloured edges
    exist that connect the partitions (e.g blue edge whose source is part of partition 1 and destination is part of partition 2),
    and colouring blue an edge of minimal weight that connects the partitions.
    
    Note that every vertex in a blue edge component (a connected set of edges, all of which are coloured blue) must 
    belong to the same partition, or else we introduce blue edges connecting the partitions.

    Args:
        graph (Graph): the graph to perform the red rule on.
        start_vertex_name (str): The graph to start the partition search on. Defaults to some arbitrary vertex.
        blue_colour (str) : the choice of colour to use to identify edges as "blue". Defaults to blue.
        colour_edge (bool) : whether to colour the selected edge blue in the graph or not. Defaults to True.
     

    Returns: a 4-tuple, where the first element is the edge of minimum weight that was selected, 
    the second element is the set of all edges that go inbetween the partitions and the third and fourth 
    elements are the vertex partitions themselves. If no valid partition set exists (no edges inbetween the partitions),
    the first 2 elements will be None.
    """

    # Store all explored vertices - that is vertices belonging to one partition or another
    explored = set([])

    # These are the partitions of the graph that we will place all the vertices into    
    partitions = [set([]),set([])]

    # As every vertex contained in a blue-edge component must belong to the same partition
    # We will call DFS to search through every blue-edge component and add all its vertices to the desired partition
    def blue_rule_dfs(current_vertex_name : str, partition_number : int = 0) -> None:
         
        # Add to explored to avoid re-exploring and add it to the corresponding partition
        explored.add(current_vertex_name)

        partitions[partition_number].add(current_vertex_name)
        
        # If already assigned to a partition, override as all blue edge components must belong to the same partition
        if current_vertex_name in partitions[(partition_number+1) % 2]: 
            partitions[(partition_number+1) % 2].remove(current_vertex_name)
        
        partition_mapping[current_vertex_name] = partition_number
        
        # Get the vertex itself so we can inspect its neighbours
        current_vertex = graph.get_vertex(current_vertex_name)
        neighbours = current_vertex.connected_vertices()
        
        for neighbour in neighbours:
            
            # Only navigate through edges which are also blue and unexplored to add them to the same partition
            edge = graph.get_edge(current_vertex_name, neighbour.name)
            
            if neighbour.name not in explored and edge.colour == blue_colour:
                blue_rule_dfs(neighbour.name, partition_number)
        
    # Pick arbitrary start vertex to search from
    if start_vertex_name is None: start_vertex_name = graph.vertices()[0]
        
    # Now we will perform a breadth-first-search to put each vertex in a partition
    
    # Reset the explored vertices as we will do a BFS - call it explored2 to avoid discarding the original, which is used for 
    # blue edge components (so we need 2 explored components, one for normal vertices, others for blue edge components)
    explored2 = set([start_vertex_name])
    explored = set([])
    
    # Store visiting nodes - used to search through BFS
    visiting = deque([start_vertex_name])
    
    # This mapping contains the partition index of each partition
    partition_mapping = { start_vertex_name : 0 }
    
    # Start by adding the first starting vertex into the first partition
    partitions[0].add(start_vertex_name)
    
    # Store the set of all vertices that are adjacent to blue-coloured edges
    blue_edge_vertices = set([])
    
    # For each vertex check if it is a blue edge vertex
    for vertex in graph.vertices(objects=True):
        for neighbour in vertex.connected_vertices():
            if graph.get_edge(vertex.name, neighbour.name).colour == blue_colour:
                blue_edge_vertices.add(vertex.name)
    
    while len(visiting) != 0:
          
        # Get the current vertex from the queue to explore
        current_vertex_name = visiting.popleft()
        current_vertex = graph.get_vertex(current_vertex_name)
        
        # Get all outgoing edges so we can find the MST
        neighbours = current_vertex.connected_vertices()
        
        for neighbour in neighbours:
            
            # Get the colour of this edge, will determine how we interpret the edge
            edge_colour = graph.get_edge(current_vertex_name, neighbour.name).colour
            
            # If the edge is blue, perform DFS so all blue edge components go to the same mapping
            if edge_colour == blue_colour and neighbour.name not in explored: 
                blue_rule_dfs(current_vertex_name, partition_mapping[current_vertex_name])
            
            # If we've already explored it then don't go down this path
            if neighbour.name not in explored2:
                    
                # If not already part of a partition, we will give it a partition opposite to the current vertex
                if partition_mapping.get(neighbour.name) is None:
                    
                    new_partition_number =  ( partition_mapping[current_vertex_name] + 1 ) % 2
                      
                    partition_mapping[neighbour.name] = new_partition_number
                    partitions[new_partition_number].add(neighbour.name)
                

                # Add to explored and visited so we will explore all its neighbours as well
                visiting.append(neighbour.name)
                explored2.add(neighbour.name)
                    
                


    # We need to select all edges which go between P1 and P2
    inbetween_edges = [edge_name for edge_name in graph.edges() 
                       if edge_name[0] in partitions[0] and edge_name[1] in partitions[1]]
     
     
    # If inbetween edges exist, otherwise the blue rule cannot be applied 
    if len(inbetween_edges) != 0:
    
        # Get all the associated weights with each edge
        inbetween_edges_and_weights = [(edge_name, graph.get_edge(*edge_name).weight) for edge_name in inbetween_edges]
        
        # Get the edge of minimal weight    
        min_weight_edge, _ = min(inbetween_edges_and_weights , key = lambda x : x[1])
        
        # Colour the minimal weight edge blue
        if colour_edge: graph.get_edge(*min_weight_edge).set_colour(blue_colour)
        
        # Give back the minimum weight edge, the inbetween edges and the two partitions
        return (min_weight_edge, inbetween_edges, *partitions)
    
    # Otherwise, the blue rule cannot be applied - just give back the partitions
    else: return (None, None, *partitions)
    
    
    

    
    
def mst_meta_algorithm(GA, red_colour : str = "red", blue_colour : str = "turquoise", 
                       generic_highlight_colours : list[str] = ["gold","magenta", "lime"]) -> tuple[float, dict[tuple[str,str], str]]:
    
    """
    Perform the minimum spanning tree meta-colouring algorithm on a graph.

    Every edge in the graph will either be designated red or blue according to the red and blue rules respectively.
    
    The red rule identifies a cycle containing no existing red edges and colours red an edge in the cycle with maximal weight.
    The blue rule identifies a vertex partition where no existing blue edges connect the partitions, and colours blue
    an edge connecting the partitions of minimal weight.

    Args:
        GA: The graph algorithm structure to perform the graph algorithm on.
        red_colour (str): what colour to use for red edges. Defaults to red.
        blue_colour (str): what colour to use for blue edges. Defaults to turquoise.
        generic_highlight_colours (list[str]): other colours for use in highlighting key vertices and edges.

    Returns: A tuple where the first element is the size of the minimum spanning tree and the second is a mapping 
    of edges to colours (either red or blue). 

    """
    
    G : Graph = GA.get_current_graph()
    
    # Remove highlighting and this could be confusing
    G.clear_highlighting()
    
    # Store original edge colours as these will all be modified throughout the algorithm
    original_edge_colours = G.get_edge_colours()
    original_textcolours = G.get_edge_textcolours()
    
    # Store the original weights of the graph so we can set them back afterwards
    original_weights = { edge_name : G.get_edge(*edge_name).weight for edge_name in G.edges()  }
    
    # Validate the graph by checking that every edge has a weight
    for edge_name in G.edges():
        edge = G.get_edge(*edge_name)
        
        # If not possible to make each edge have numeric weight then set it to have 0 weight
        if isinstance(edge.weight, str):
            
            # Try to make it into an integer
            try:
                if "." in edge.weight:
                    edge.set_weight(float(edge.weight))
                else:
                    edge.set_weight(int(edge.weight))
                    
            # If not a valid instance then just set to 0
            except: edge.set_weight(0)
        
        elif not isinstance(edge.weight, (int, float, number)):
            edge.set_weight(0)
    
    G.annotate(f"We will find a minimum spanning tree for {G.name} using the edge-colouring meta algorithm.")
    GA.save_state()
    
    # Store the total weight of all red and blue edges for documentation
    blue_edge_weights = 0
    red_edge_weights = 0
    
    # Store the colours of each edge
    edge_colours = {}
    
    def perform_red_rule() -> int:
        nonlocal red_edge_weights
        
        # Apply the red rule to get all the relevant data
        maximal_edge, cycle_edges = red_rule(G, red_colour=red_colour)
    
        # If the red rule cannot be applied we get None
        if maximal_edge is not None:
    
            # The weight of the maximal weight edge            
            edgeweight = G.get_edge(*maximal_edge).weight
            
            # Update the total red edge weight
            red_edge_weights += edgeweight
            
            # This puts the edge in a readable string format
            strformat = maximal_edge[0] + "-" + maximal_edge[1]
            
            # Perform annotations and highlights to show the red rule being applied
            
            # Highlight all edges involved in the cycle so we can see all the edges compared to the maximal edge
            for edge in cycle_edges:
                if edge != maximal_edge:
                    G.highlight_edge(edge, generic_highlight_colours[0]) 
            
            # Explain what's happening to the reader
            G.annotate(f"Red rule: identify a simple cycle in {G.name} with no red edges and colour the maximal weight edge {strformat} ({edgeweight}) red.")
            G.highlight_edge(maximal_edge, red_colour)
            
            # Now we will record that this edge is red - we will use this in the algorithm text
            edge_colours[maximal_edge] = "red"
            edge_colours[maximal_edge[::-1]] = "red"
            
            # Update the text
            GA.clear_text()
            GA.add_text(algorithm_text())
            
            GA.save_state()
            
            # Remove all highlights so we can proceed with the next rule to use
            G.clear_highlighting()
            
            # This hash will be either 0 or 1, 0 = apply red rule again, 1 = apply blue rule
            return hash(maximal_edge) % 2
            
        else:
            # If it's no longer possible to apply the red rule then we return -1 to show this
            G.annotate(f"There are no more applications of the red rule that can be applied to {G.name}.")
            GA.save_state()
            return -1
            
    def perform_blue_rule() -> int:
        nonlocal blue_edge_weights
        
        # Perform the blue rule as needed
        minimal_edge, _, P1, P2 = blue_rule(G, blue_colour=blue_colour)
        
        # Highlight the partition vertices
        for vertex in P1: G.highlight_vertex(vertex, generic_highlight_colours[1])
        for vertex in P2: G.highlight_vertex(vertex, generic_highlight_colours[2])
        
        if minimal_edge is not None:
            
            # The weight of the minimal weight edge
            edgeweight = G.get_edge(*minimal_edge).weight
            
            # Update the total weight of all the blue edges
            blue_edge_weights += edgeweight
            
            # Again put the edge in string format 
            strformat = minimal_edge[0] + "-" + minimal_edge[1]
            
            # Explain what's happening and highlight the minimal weight edge blue
            G.annotate(f"Blue rule: identify a vertex partition in {G.name} where no existing blue edges connect the partitions, and colour the minimal weight edge {strformat} ({edgeweight}) blue. ")
            G.highlight_edge(minimal_edge, blue_colour)
            
            # We record that these edges are now blue - useful for the algorithm text
            edge_colours[minimal_edge] = "blue"
            edge_colours[minimal_edge[::-1]] = "blue"
            
            # Update the text
            GA.clear_text()
            GA.add_text(algorithm_text())
            
            GA.save_state()
            
            # Remove the highlighting so we can continue the next rule
            G.clear_highlighting()
            
            # If 0, we use the red rule, if 1 we re-use the blue rule again
            return hash(minimal_edge) % 2
        
        else:
            
            # If it's no longer possible to apply the blue rule then we return -1 to show this
            G.annotate(f"There are no more partitions of {G.name} that don't contain blue edges. Here is an attempted partition.")
            GA.save_state()
            
            # Remove the partition colours so we can apply the red rule now
            G.clear_highlighting()
            
            return -1
    
    def algorithm_text() -> str:
        # Get all red and blue edges in a string format (we only want A-B, not B-A so we do an arbitrary exclusion a <= b)
        red_edges = (str([ f"{edge_name[0]}-{edge_name[1]}" 
                     for edge_name, colour in edge_colours.items() if colour == "red" and edge_name[0] <= edge_name[1]])
        .replace("[","{").replace("]","}").replace("'",""))
        
        blue_edges = (str([ f"{edge_name[0]}-{edge_name[1]}" 
                     for edge_name, colour in edge_colours.items() if colour == "blue" and edge_name[0] <= edge_name[1]])
        .replace("[","{").replace("]","}").replace("'",""))
        
        uncoloured_edges = (str([ f"{edge_name[0]}-{edge_name[1]}" 
                     for edge_name in G.edges() if edge_colours.get(edge_name) is None and edge_name[0] <= edge_name[1]])
        .replace("[","{").replace("]","}").replace("'",""))
        
        # This is all of the information that we will provide
        return f"Total blue edge weight: {blue_edge_weights}\nBlue edges: {blue_edges}\nTotal red edge weight: {red_edge_weights}\nRed edges: {red_edges}\nUncoloured edges: {uncoloured_edges}"
    
    # Our initial choice of using a red rule or blue rule can be made by hashing
    # This simulates non-deterministic choice while actually remaining deterministic 
    blue_rule_chosen = hash(G.name) % 2
    
    # This list stores whether the red and blue rules respectively have been exhausted and cannot be used again
    # If true, exhausted, if false, then still applicable
    exhausted = [False,False]
    
    # While there are still edges that haven't been assigned a colour:
    while len(edge_colours) != len(G.E):
        
        # Choose the blue rule unless the red rule is exhausted
        if ( blue_rule_chosen and not exhausted[1]) or exhausted[0]:
            result = perform_blue_rule()
            if result == -1: 
                exhausted[1] = True
                result = 0

        # Otherwise choose the red rule
        else:
            result = perform_red_rule()
            if result == -1: 
                exhausted[0] = True
                result = 1
        
        # Now determine if we want to use the blue rule or the red rule from the result
        blue_rule_chosen = result
        
    
            
    G.annotate(f"We have now fully coloured {G.name} - all blue edges form a minimum spanning tree (MST) of total size {blue_edge_weights}.")
    GA.save_state()
    
    # We hide all the red edges so we see the MST only
    for edge_name in G.edges():
        edge = G.get_edge(*edge_name)
        
        if edge.colour == red_colour:
            edge.set_colour(G.background_colour)
            edge.set_textcolour(G.background_colour)
            
    GA.save_state()
    
    # Restore the original edge colours as we changed them all
    G.assign_edge_colours(original_edge_colours)
    G.assign_edge_textcolours(original_textcolours)
    
    # Remove all text and supporting information to restore the original graph
    G.clear_annotations()
    G.clear_highlighting()
    GA.clear_text()
    
    # Restore the original weights back to their intended values
    for edge, weight in list( original_weights.items() ):
        G.get_edge(*edge).set_weight(weight)        
        
    return (blue_edge_weights, edge_colours)