from graph import Graph
from helper_functions import get_n_colours
from algorithms.dfs import dfs


def bcc_algorithm(GA, start_vertex_name : str = None, highlight_colour : str = "gold", 
                  vertex_colours : str = ["red","lime", "brown"]) -> tuple[set[str], list[list[tuple[str,str]]]]:
    
    """
    Given a connected undirected graph, determine its articulation points and biconnected components.
    
    An articulation point is a vertex which, when removed, disconnects the graph. 
    A biconnected component is an equivalence class of edges in the equivalence relation where a cycle exists between
    any pair of endpoints of the vertex edges.

    Args:
        GA: the graph algorithm that handles this algorithm.
        start_vertex_name (str): the name of the vertex to begin DFS at.
        highlight_colour (str): the highlight colour of the DFS tree.
        vertex_colours (list[str]) - the list of colours for vertices, denoting articulation points, leaf nodes
        and vertices that have been fully explored respectively.

    Raises:
        Exception: graph is not connected
        Exception: graph is directed; edge (A,B) exists but not (B,A)

    Returns:
        tuple: a pair, where the first element is the set of articulation points and the second is the list
        of all biconnected components, where components[i] = component i+1. E.g components[3] = the 4th biconnected component.
    """
    
    # This is the graph we will perform the algorithm on
    G : Graph = GA.get_current_graph()
    
    # Store the original colours so we can put them back later    
    original_vertex_colours = G.get_vertex_colours()
    original_edge_colours = G.get_edge_colours()
    
    # Remove existing highlighting as it can become confusing
    GA.clear_highlighting(G)
    
    # Get the number of SCCs - if this number > 1 then the graph is invalid
    # We can't form a biconnected equivalence relation if the graph isn't connected
    connected = len(GA.run_algorithm(dfs, graph=G, capture=False)[2]) == 1
    
    # Raise an exception if graph is not connected
    if not connected: raise Exception(f"graph {G.name} is not connected - graph must be connected to run this algorithm")
    
    # Enforcing bidirectional edges - no single directed edges are permitted - must be an undirected graph
    edgeset = set(G.edges())
    for edge in edgeset:
        if edge[::-1] not in edgeset:
            raise Exception(f"Edge {edge} exists in {G.name} but not {edge[::-1]} - graph must be undirected to run this algorithm")
    
    # Determine the number of colours to assign for our BCC graph
    # Of course we have to know this ahead of time, so if capturing we pre-run the algorithm to find out how many
    # Maximum V colours so use that as an upper bound in case we're not capturing
    num_colours = len(G.V)
    if GA.capturing:
        # Pre-run the algorithm to determine how many colours we need (and hence how many BCCs)
        num_colours = len(GA.run_algorithm(bcc_algorithm, graph=G, capture=False)[1])
        
    # Get as many colours as we need for colouring in the BCCs
    colours = get_n_colours(num_colours)
    
    # If no start vertex provided, begin with arbitrary vertex
    if start_vertex_name is None: start_vertex_name = G.vertices()[0]
    
    # Keep track of all explored vertices for the DFS algorithm
    explored = set([start_vertex_name])
    
    # These are the colours we will use to denote leaf nodes and articulation points
    # Leaf nodes and articulation point vertices are mutually exclusive, and finish colours used for non-articulation vertices
    leaf_node_colour, finish_colour, articulation_colour = vertex_colours
    
    # Store articulation points here
    articulation_points = set([])

    # We store the level of each vertex here - used for determining lowpoints
    levels = {start_vertex_name : 0}
    
    # Store low points so we can show them
    low_points = {start_vertex_name : 0}
    
    # This is the text we show when determining the articulation points
    def algorithm_text() -> str:
        articulation_points_str = str(articulation_points).replace("'","") if len(articulation_points) !=0 else "{}"
        
        levels_lowpoints = { vertex : (low_points.get(vertex,"?"), levels.get(vertex,"?")) for vertex in G.vertices() }
        levels_lowpoints_str = str(levels_lowpoints).replace("'","")
        
        textstring = f"Articulation points: {articulation_points_str}\nVertex low-points and levels: {levels_lowpoints_str}"
        
        return textstring
    
    # This is the text we show when determining the biconnected components
    def algorithm_text2() -> str:
        bccs_str = (str({ f"BCC {i+1}" : [source +"-" +dest for source,dest in arr] for i, arr in enumerate(bccs) } )
                    .replace("'","").replace("[","{").replace("]","}"))
        
        return bccs_str
        

    # This function calculates the levels and low points of each vertex, and hence determines articulation points
    def find_articulation_points(current_vertex, level) -> int:
        
        # Update the levels and explored so future nodes can use this information
        explored.add(current_vertex.name)
        levels.update({current_vertex.name : level})
        
        # Every vertex starts with its low point being its level, as it is assumed reachable by its children
        low_point = level
        
        # A leaf node is a node with no unexplored vertices
        # This will remain true until a neighbour is found without unexplored vertices
        is_leaf_node = True

        # Don't forget to update the text of articulation points
        GA.clear_text()
        GA.add_text(algorithm_text())

        neighbours = current_vertex.connected_vertices()
        
        for neighbour in neighbours:
            if neighbour.name not in explored:
                # We've found an unexplored vertex so it cannot be a leaf node
                is_leaf_node = False
                
                # Highlight this to show the next step of the DFS search
                G.highlight_edge((current_vertex.name, neighbour.name), "gold")
                G.highlight_vertex(neighbour.name, "gold")
                
                # Update the levels here - this is inefficient but it lets us show the levels on time
                levels.update({neighbour.name : level+1})
                GA.add_text(algorithm_text())
                
                G.annotate(f"Visit {current_vertex.name}'s neighbour {neighbour.name} (level {level+1})")
                GA.save_state()
                
                # Recursively find the articulation points of each child of this vertex to check if it's a violation
                neighbour_lowpoint = find_articulation_points(neighbour, level+1)
                low_point = min(low_point, neighbour_lowpoint)
                
                # This is the main condition on which this algorithm is based for finding articulation points
                if neighbour_lowpoint >= level:
                    articulation_points.add(current_vertex.name)
                    low_points.update({current_vertex.name : neighbour_lowpoint})
                    
                    G.annotate(f"{neighbour.name} has a low-point of {neighbour_lowpoint} >= {level} (level of {current_vertex.name}), so {current_vertex.name} must be an articulation point.")

                    GA.add_text(algorithm_text())
                    current_vertex.set_colour(articulation_colour)      
                    GA.save_state()          
            else:
                # If it's already explored then we still need to check it doesn't have a lower low point
                # Or else we wouldn't be following the definition of low points
                low_point = min(low_point, levels.get(neighbour.name))
        
        if is_leaf_node:
            low_point = min([ levels.get(neighbour.name) for neighbour in neighbours ])     
            
            # Update here as well - inefficient to update twice but we need to show it on the algorithm text in real time
            low_points.update({ current_vertex.name : low_point })
            G.annotate(f"{current_vertex.name} is a leaf node whose low-point is {low_point}.")
            GA.add_text(algorithm_text())
            current_vertex.set_colour(leaf_node_colour)
            GA.save_state()
        
        # We also annotate for regular vertices whose levels and low points are found
        # Don't do this for articulation points since they've already been commented on        
        elif current_vertex.name not in articulation_points:
            # Add to the low points so we can update the text            
            low_points.update({ current_vertex.name : low_point })
            GA.clear_text()
            GA.add_text(algorithm_text())
            G.annotate(f"{current_vertex.name} has a level of {level} and a low-point of {low_point}.")
            current_vertex.set_colour(finish_colour)
            GA.save_state()
        
        # Give the lowpoint so we can recursively determine the lowpoints of other vertices
        return low_point
    
    # Now that we have the articulation points, we find the biconnected components
    def find_biconnected_components(current_vertex) -> None:
        
        # We're currently looking at this vertex to add edges to the biconnected equivalence relation
        explored.add(current_vertex.name)
        
        neighbours = current_vertex.connected_vertices()
        
        if current_vertex.name not in articulation_points:
        
            # Get the index of the current biconnected component to add to
            bcc_index = len(bccs) - 1
            
            # Colour every neighbour with this biconnected component
            for neighbour in neighbours:
                
                # We have to store both because it's an undriected graph
                edge_name = (current_vertex.name, neighbour.name)
                reverse_edge_name = (neighbour.name, current_vertex.name)
                
                # Don't add duplicate edges
                if edge_name not in coloured_edges:
                
                    edge = G.get_edge(*edge_name)
                    
                    # Highlight and colour this edge to show that it belongs to this equivalence class
                    edge.set_colour(colours[bcc_index])
                    G.highlight_edge(edge_name, colours[bcc_index])
                    
                    # Add to the edge so we have coloured edges
                    bccs[bcc_index].append(edge_name)
                    coloured_edges.add(edge_name)
                            
                    
                    # If bidirectional then add the other way as well
                    if edge.is_bidirectional():
                        bccs[bcc_index].append(reverse_edge_name)
                        coloured_edges.add(reverse_edge_name)

            GA.clear_text()
            GA.add_text(algorithm_text2())
            G.annotate(f"Add all edges of vertex {current_vertex.name} to biconnected component {bcc_index+1}.")
            GA.save_state()
            
            # Now recursively exploer all neighbours that aren't articulation points
            for neighbour in neighbours:
                if neighbour.name not in explored and neighbour.name not in articulation_points:
                    find_biconnected_components(neighbour)
        

    # Get the neighbours so we can begin the DFS search
    start_vertex = G.get_vertex(start_vertex_name)
    start_vertex_neighbours = start_vertex.connected_vertices()
    
    # Cosmetic changes and presentation
    G.annotate(f"To find the biconnected components of {G.name}, we must first find its articulation points ({articulation_colour}). Leaf nodes will be coloured {leaf_node_colour} and finish nodes will be coloured {finish_colour}.")
    GA.save_state()
    
    G.highlight_vertex(start_vertex_name, highlight_colour)
    G.annotate(f"Start the DFS search at vertex {start_vertex_name}.")
    GA.add_text(algorithm_text())
    GA.save_state()
    
    # The root of the search tree is an articulation point iff it's connected to 2 or more vertices
    unexplored_vertex_count = 0
    for neighbour in start_vertex_neighbours:
        if neighbour.name not in explored:
            
            # Cosmetic changes and presentation
            G.highlight_edge((start_vertex_name, neighbour.name), highlight_colour)
            G.highlight_vertex((neighbour.name), highlight_colour)
            G.annotate(f"Now explore {neighbour.name} from {start_vertex_name}.")
            
            # Update the levels here so we can see the change in the algorithm text in real time
            # A little inefficient but only by a constant modifier, and the focus is presentation
            levels.update({neighbour.name : 1})
            GA.add_text(algorithm_text())
            GA.save_state()
            
            # Now recursively check this subtree for articulation points
            find_articulation_points(neighbour, 1)
            
            # We have explored another vertex from the start - if we get to 2 or more then root becomes an artic. point
            unexplored_vertex_count += 1
            
    if unexplored_vertex_count > 1:
        G.annotate(f"{start_vertex_name} is connected to {unexplored_vertex_count} vertices in the DFS tree, so must be an articulation point.")
        articulation_points.add(start_vertex_name)
        GA.add_text(algorithm_text())
        start_vertex.set_colour(articulation_colour)
    else:
        G.annotate(f"{start_vertex_name} has only one edge in the DFS tree, so cannot be an articulation point.")
        start_vertex.set_colour(finish_colour)
        
    GA.save_state()
        
    G.clear_highlighting()
    
    G.annotate(f"We have found all articulation points of {G.name} ({articulation_colour}) - we can now find the biconnected components.")
    
    GA.save_state()
    
    # We represent biconnected components as a mapping from integers to vertex sets
    # Each element is a list of vertices, and its "key" is the list index
    # The inner list stores the first biconnected component edges
    bccs = []
    
    # Reset explored vertices for bcc discovery    
    explored = set([])
    
    # We keep track of all edges that have been coloured to avoid colouring twice
    coloured_edges = set([])
    
    # Keep exploring until we find all the biconnected components
    while len(explored) != len(G.V): 
        
        # Next vertex to start checking for BCCs on
        next_vertex_name = list((set(G.vertices()) - explored))[0]
        
        next_vertex = G.get_vertex(next_vertex_name)
        
        # Otherwise we have a new biconnected component (assuming not artic. point)
        if next_vertex_name not in articulation_points: bccs.append([])
        
        find_biconnected_components(next_vertex)
    
    # Finally the last case is edges between two articulation points in the graph
    for edge_name in G.edges():

        # Check to see both source and destination are artic. points not already assigned, otherwise already done
        if edge_name[0] in articulation_points and edge_name[1] in articulation_points and edge_name not in coloured_edges:
            
            # Retrieve the edge itself
            edge = G.get_edge(*edge_name)

            # Then we have a new BCC containing this edge
            bccs.append([edge_name])
            bcc_index = len(bccs) - 1
            
            # Add to the list of coloured edges so we don't add it again
            coloured_edges.add(edge_name)
            
            # Don't forget to also add the other edge if we're bidirectional
            if edge.is_bidirectional(): 
                bccs[bcc_index].append(edge_name[::-1])
                coloured_edges.add(edge_name[::-1])
            
            # Set colour and highlight to show new colour
            edge.set_colour(colours[bcc_index])
            edge.highlight(colours[bcc_index])
            G.annotate(f"The edge {edge_name[0]}-{edge_name[1]} is between two articulation points, forming its own BCC.")
            GA.add_text(algorithm_text2())
            GA.save_state()
            

    
    GA.add_text(algorithm_text2())
    G.annotate(f"We have now found all {len(bccs)} biconnected components (BCCs) of {G.name}.")
    
    GA.save_state()
    # Also give another version without the highlights
    G.clear_highlighting()
    GA.save_state()
    
    # Reset everything in the graph back to how it was originally
    GA.clear_text()
    G.clear_annotations()
    G.assign_vertex_colours(original_vertex_colours)
    G.assign_edge_colours(original_edge_colours)
    
    # Return the articulation points and the biconnected components equivalence relation
    return (articulation_points, bccs)

        
        