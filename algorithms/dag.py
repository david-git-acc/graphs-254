from algorithms.dfs import dfs
from helper_functions import paragraphise

# Provide a topological sort of the graph, specifying whether it is a DAG and the violating edges of the sort
# Returns a 3-tuple, where the first element is True/False for whether it's a DAG
# The second element is the assignment of numbers to vertices (given by vertex names)
# The third element is a list of edges in the graph, if any, which violate the topological sorting
# fail and success colour are used to check if an edge follows or does not follow the topological sort
# We use the DFS-based algorithm to perform the topological sort
def topological_sort(GA, start_vertex_name = None, highlight_colour : str = "gold", finish_colours : list[str] = ["red","lime"]
                     , fail_colour : str = "red", success_colour : str = "lime" ) -> tuple[bool, dict[str, int], list[tuple[str,str]] ]:
    
    def algorithm_text(ordering : dict[str,int], violating_edges : set[tuple[str,str]], cpl : int) -> str:
        
        # This is the string containing the topological ordering of the graph
        ordering_string = str(ordering).replace("'","")
        
        # Store the violating edges string for edges that do not conform to the topological sort
        violating_edges_str = str(violating_edges).replace("'","") if len(violating_edges) != 0 else "{}"
        
        # Give back the ordering of vertices and the edges in string format
        return paragraphise("Topological ordering: " + ordering_string + "\nViolating edges: " + violating_edges_str, cpl)
               
    # Switch to the current graph being used by the graph algorithm
    G = GA.get_current_graph()
    
    # Remember the current assignment of colours so we can update them back once we've finished DFS
    current_colours = GA.get_vertex_colours(G)
    
    GA.annotate(G, f"To determine if {G.name} is a DAG, we will first need the DFS finish numbers on {G.name}.")
    GA.save_state()
    
    # Run DFS on the graph to get the DFS finish numbers
    dfs_finish_numbers,_ = GA.run_algorithm(dfs, graph=G, start_vertex_name=start_vertex_name, highlight_colour=highlight_colour, 
                                          finish_colours=finish_colours, kill_existing=False, capture=GA.capturing, save_video = False)
    
    # The topological ordering of G = number of vertices - finish_number + 1, so we need the number of vertices n
    n = len(G.V)
    
    # Compute the topological ordering from the finish numbers using the formula
    topological_ordering = { vertex : n - finish_number + 1 for vertex, finish_number in dfs_finish_numbers.items()  }
    
    # Store the set of edges which violate the topological sort
    violating_edges = set([])
    
    # Remove all the colourful decorations made by DFS
    GA.clear_text()
    GA.clear_annotations(G)
    GA.annotate(G, f"We have obtained the DFS finish numbers for {G.name} and thus have obtained a topological sort." )
    
    # We will replace all vertex names so that they contain their topological orderings
    # Hence we need to store these names somewhere so that we can restore the original names afterwards
    original_vertex_names = G.vertices()
    new_vertex_names = [ vertex_name + f" ({topological_ordering[vertex_name]})" for vertex_name in original_vertex_names ]
    
    # Create mappings between the original and new vertex names so we can use them
    org_to_new = dict(zip(original_vertex_names, new_vertex_names))
    new_to_org = dict(zip(new_vertex_names, original_vertex_names))
    
    # Re-establish the original vertex colours
    G.assign_vertex_colours(current_colours)
    
    # Rename the vertices to the new names
    GA.rename_vertices(G, org_to_new)
    
    # Add the text to show the topological ordering
    GA.add_text(algorithm_text(topological_ordering, violating_edges, G.characters_per_line))
        
    # Save at this point before we begin checking edges - do it twice so we get more time to look at it
    GA.save_state()
    GA.save_state()
    
    # Store a list of edges that will need to be highlighted (because they follow the ordering)
    non_violating_edges = []

    # Now for each edge we will check if it satisfies the definition
    for source_name, dest_name in G.edges():
        
        # Get the original names so we can compare them
        old_source_name, old_dest_name = ( new_to_org[source_name], new_to_org[dest_name] )
        
        # If the topological ordering fails, we have a "bad" edge
        if topological_ordering[old_source_name] >= topological_ordering[old_dest_name]:
            
            # Highlight this edge to show that it breaks the ordering
            GA.highlight_edge(G, (source_name, dest_name), fail_colour)
            
            # Add the violating edge to the set of violating edges
            violating_edges.add((old_source_name, old_dest_name))
            
            # We will need to update the upper text and annotations to account for this violating edge
            GA.clear_text()
            GA.add_text(algorithm_text(topological_ordering, violating_edges,G.characters_per_line))
            GA.annotate(G, f"The edge from {source_name} to {dest_name} violates the topological ordering.")
            GA.save_state()
            
        # If they do follow the ordering then add them to the list of non violating edges for later highlighting
        else: non_violating_edges.append((source_name, dest_name))
            
    # For every edge that passes the check we colour it in the success colour
    for edge_name in non_violating_edges: 
        # If the edge isn't already highlighted (see: bidirectional edges)
        if G.highlighted_edges.get(edge_name) is None:
            GA.highlight_edge(G, edge_name, success_colour)

    # This allows us to determine if G is a DAG - if all edges follow the ordering then it's a DAG
    is_dag = len(violating_edges) == 0
    
    # Now we need to give our concluding statement
    if is_dag:
        GA.annotate(G, f"As all edges in {G.name} follow the topological ordering, {G.name} is a DAG.")    
    else:
        GA.annotate(G, f"As some edges in {G.name} violate the topological ordering, {G.name} is not a DAG.")

    # Now save the state as it's the concluson
    GA.save_state()
    
    # Get rid of the highlighting now that we don't need it anymore
    GA.clear_highlighting(G)
    
    # Give the vertices their old names back
    GA.rename_vertices(G, new_to_org)
    
    # Also get rid of any text we added to return to the original graph
    GA.clear_text()
    GA.clear_annotations(G)
     
    # Give back all the important information
    return (is_dag, topological_ordering, list(violating_edges))
    
    
    
    