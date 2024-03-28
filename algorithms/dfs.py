

from helper_functions import paragraphise

# Depth-first-search that computes vertex finish numbers
def dfs(GA, start_vertex_name = None, highlight_colour : str = "gold", finish_colours : list[str] = ["red","lime"],
        capture : bool = True) -> dict[str,int]:
    
    # Create mappings to convert from the old vertex names (the vertex names themselves) 
    # to the new vertex names which are the vertex + (DFS score)
    org_to_new_names = {}
    new_to_org_names = {}
    
    # Text to output during the algorithm execution
    def algorithm_text() -> str:
        
        # We will show the explored vertices - remove quotation marks to make it look like a real set
        explored_str : str = str(explored).replace("'","") if len(explored) != 0 else "{}"
        
        # Get the finish numbers as well, we will show this at each step
        finish_numbers_str : str = str(finish_numbers).replace("'","")
        
        textstring = "Explored: " + explored_str + "\nFinish numbers: " + finish_numbers_str
        
        return paragraphise(textstring, G.characters_per_line)
    
    # Once all paths out of a vertex have been explored we have to execute some code to show this in the animation
    def finish_vertex(GA, current_vertex) -> None:
        
        # Explain the finish number of the current vertex
        current_vertex.annotate(f"All paths out of {current_vertex.name} have been explored ({finish_colours[1]}), so its finish number is {current_finish_number}.")
        
        # Set its colour to the fully finished colour
        current_vertex.set_colour(finish_colours[1])
        
        # We will put the finish number score of the vertex as its new name so we can see it in the graph
        new_vertex_name = current_vertex.name + f" ({current_finish_number})"
        
        # Update the name mappings so we know what the original name of the vertex is
        org_to_new_names.update({ current_vertex.name : new_vertex_name  })
        new_to_org_names.update({ new_vertex_name : current_vertex.name })
        
        # Rename the vertex so that it also includes its DFS finish number
        current_vertex.rename(new_vertex_name)
        
        # Update the text for the plot
        GA.clear_text()
        GA.add_text(algorithm_text())
        GA.save_state()
    
    def recursive_dfs(GA, current_vertex) -> None:
        nonlocal current_finish_number
        
        # Since we're at this vertex, we must've explored it        
        explored.add(current_vertex.name)
        
        # Update the text for the plot
        GA.clear_text()
        GA.add_text(algorithm_text())
        GA.save_state()
        
        # Get all the adjacent neighbours to this vertex (both = don't count incoming edges)
        neighbours = current_vertex.connected_vertices(both=False)
        
        # Recursively explore every vertex, updating the finishing numbers as we go
        for neighbour in neighbours:
            # Only look at this vertex if we've not explored it before otherwise endless loop
            if neighbour.name not in explored and new_to_org_names.get(neighbour.name,"") not in explored: 
                
                # Highlight the edge and vertices to show that they've been explored
                GA.highlight_edge(G,(current_vertex.name, neighbour.name), highlight_colour)
                GA.highlight_vertex(G,neighbour.name, highlight_colour)
                current_vertex.annotate(f"Visit {current_vertex.name}'s child vertex {neighbour.name}.")
                
                recursive_dfs(GA, neighbour)
                
                
        # Once we've explored every possible path from this vertex, we've finished - increment the finish number
        current_finish_number += 1
        finish_numbers[current_vertex.name] = current_finish_number
        
        finish_vertex(GA, current_vertex)
    
    # Fetch the current graph in use by the graph algorithm
    G = GA.get_current_graph()
    
    # Get the original set of vertex names - we will use this to find new vertices to start at after exhausting a search
    original_vertices = set(G.vertices())
    
    # Get the original colours of the graph so we can reset them at the end
    original_colours = G.get_vertex_colours()
    
    # If no start vertex, select arbitrary vertex
    if start_vertex_name is None: start_vertex_name = G.vertices()[0]
    
    # Starting vertex for our DFS
    start_vertex = GA.get_vertex(G, start_vertex_name)
    
    # Maintain a set of all explored vertices to avoid recursion
    # The starting vertex is trivially always explored
    explored = set([start_vertex_name])
    
    # Initialise empty finish numbers for all vertices
    finish_numbers = { vertex : 0 for vertex in G.vertices() } 
    
    # This will be incremented by the recursive instances to calculate the DFS finish numbers
    current_finish_number = 0
    
    # Start with all vertices unexplored - mark them as red (the unexplored colour)
    GA.annotate(G, f"Initialise all vertices as unexplored and unfinished ({finish_colours[0]}).")
    GA.set_vertex_colour(G, finish_colours[0])
    
    GA.highlight_vertex(G,start_vertex_name, highlight_colour)
    start_vertex.annotate("Start our DFS search at this vertex.")
    recursive_dfs(GA, start_vertex)
  
    # Keep doing this until every vertex is explored
    while len(explored) != len(G.V):
        # Pick some arbitrary unexplored vertex to restart the search from
        new_start_vertex_name = list(original_vertices - explored)[0]
        new_start_vertex = GA.get_vertex(G,new_start_vertex_name)
        
        GA.highlight_vertex(G,new_start_vertex_name, highlight_colour)
        new_start_vertex.annotate("Restart our DFS search at this vertex.")
        
        recursive_dfs(GA, new_start_vertex)
        
    # Rename the vertices back to their original names
    GA.rename_vertices(G, new_to_org_names)
    
    GA.annotate(G, f"We have now fully explored {G.name} using Depth-first-search (DFS).")
    
    # Update the text for the plot
    GA.clear_text()
    GA.add_text(algorithm_text())
    GA.save_state()
    
    # Get rid of the highlighting, we don't need it anymore
    GA.clear_highlighting(G)
    
    # Return the vertices back to their original colours
    G.assign_vertex_colours(original_colours)
    
    # Get rid of the annotations and text we added to return to the original graph
    GA.clear_annotations(G)
    GA.clear_text()

    # Give back the finish numbers once we're done
    return finish_numbers

  
        
        
        
    
