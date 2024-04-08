from collections import deque


def bfs(GA, start_vertex_name : str = None, highlight_colour : str = "gold", finish_colours : list[str] = ["red","lime"],
        persistent : bool = True, pre_explored : list[str] = [], skip_intro : bool = False) -> tuple[list[str], tuple[str, list[tuple[str, int]]],list[list[str]]]:
    
    """
    Perform a breadth-first-search (BFS) on a graph.

    Args:
        GA: The graph algorithm structure to run this algorithm on.
        start_vertex_name (str): the name of the vertex to begin the search on. Defaults to an arbitrary vertex.
        highlight_colour (str): the colour to highlight BFS tree edges with to mark the BFS forest. Defaults to gold.
        finish_colours (list[str]): the unexplored and fully explored colours respectively of a vertex. Defaults to red and lime.
        persistent (bool): Whether the search should continue after having exhausted the start vertex's connected component. Defaults to True.
        pre_explored (list[str]): the list of vertices which should be exempted from exploration in the graph. Defaults to [].
        skip_intro (bool): Whether to include the BFS introduction in the video. Useful to turn off for algorithms using BFS as a subroutine. Defaults to False.
    
    Raises:
        Exception: If the specified starting vertex is also listed as a pre-explored vertex, this is undefined behaviour.

    Returns: tuple[list[str], tuple[str, list[tuple[str, int]]],list[list[str]]]
    
        A tuple, whose first element is the explored vertices, sorted in ascending order of exploration.
        The second element is a tuple whose first element is the start vertex name, and second element is a list of tuples 
        where the first element is an explored vertex and the second element is its distance from the start vertex. This list 
        is sorted in ascending order of distance.
        The third element stores the connected components of the searched graph.
    """
    
    # Get the current graph to perform the graph algorithm on
    G = GA.get_current_graph()
    
    # Store the original colours of the graph so we can reset this later
    original_colours = G.get_vertex_colours()

    # Get the original vertex names - this will help us ensure we don't pick a vertex already explored
    original_vertex_names = list( set( G.vertices() ) - set(pre_explored))

    # If no start selected, pick arbitrarily
    if start_vertex_name is None: start_vertex_name = original_vertex_names[0]
    
    # If the user sets the start vertex to a vertex in pre-explored then this is not acceptable
    if start_vertex_name in pre_explored: raise Exception(f"Start vertex {start_vertex_name} set as already explored")
    
    # These dictionaries will map between the old vertex names and the new vertex names
    # This is because we have to rename the vertices so it shows their shortest path distance
    org_to_new = {}
    new_to_org = {}
    
    def algorithm_text() -> str:
        shortest_paths_list = sorted(list(shortest_paths.items()), key = lambda x : x[1])
        shortest_paths_str = str(shortest_paths_list).replace("'","").replace("[","{").replace("]","}")
        
        explored_str = str(explored_list).replace("'","").replace("[","{").replace("]","}")
        
        textstring = f"Explored vertices: {explored_str}\nShortest paths from {start_vertex_name}: {shortest_paths_str}"
    
        return textstring
    
    # Store the connected components of the graph
    connected_components = [[start_vertex_name]]
    
    # Store explored vertices - first is for efficient checking, second is for remembering the order
    explored = set([start_vertex_name] + pre_explored)
    explored_list = [start_vertex_name]
    
    # Store the shortest edge length paths in terms of how many edges away they are, initialised at infinity
    shortest_paths = {}
    
    # Obviously the start vertex is 0 edges away from itself
    shortest_paths[start_vertex_name] = 0
    
    # We will set every non-pre-explored vertex to the initial colour
    colour_mapping = dict(zip( list(original_vertex_names), [finish_colours[0]]*len(original_vertex_names) ))
    
    if not skip_intro:
        G.annotate(f"Initialise all vertices as unexplored ({finish_colours[0]}).")
        G.assign_vertex_colours(colour_mapping)
        GA.save_state()
    
    G.annotate(f"We will begin Breadth-First-Search (BFS) at vertex {start_vertex_name}.")
    G.highlight_vertex(start_vertex_name, highlight_colour)
    GA.clear_text()
    GA.add_text(algorithm_text())
    GA.save_state()
    
    # Perform BFS on a single connected component of the graph
    def component_bfs(component_start_vertex_name : str, component_index : int):
    
        # We keep exploring until there are no vertices left to visit
        visited = deque([component_start_vertex_name])
            
        while len(visited) != 0:
            # Get the current vertex so we can explore its neighbours
            this_vertex_name = visited.popleft()
            
            if this_vertex_name != component_start_vertex_name:
                G.annotate(f"Now we will explore vertex {this_vertex_name}.")
                GA.clear_text()
                GA.add_text(algorithm_text())
                GA.save_state()
              
            # Get the vertex itself so we can find its neighbours
            this_vertex = G.get_vertex(this_vertex_name)
            
            neighbours = this_vertex.connected_vertices()
            
            for neighbour in neighbours:
                if neighbour.name not in explored and new_to_org.get(neighbour.name,"") not in explored:
                    # Add to the appropriate storage lists
                    explored.add(neighbour.name)
                    explored_list.append(neighbour.name)
                    connected_components[component_index].append(neighbour.name)
                    visited.append(neighbour.name)
                    
                    # Update the distance for this shortest path - same path as previous edge + 1 more edge
                    shortest_paths[neighbour.name] = shortest_paths[this_vertex_name] + 1
                    
                    # Perform the edge and vertex highlighting to show the BFS tree being built at each step
                    G.highlight_edge((this_vertex_name, neighbour.name), highlight_colour)
                    neighbour.highlight(highlight_colour)
                    G.annotate(f"Visit {this_vertex_name}'s neighbour {neighbour.name}, {shortest_paths[neighbour.name]} edges away from {start_vertex_name}.")
                    GA.clear_text()
                    GA.add_text(algorithm_text())
                    GA.save_state()

            G.annotate(f"We have now explored all neighbours of vertex {this_vertex_name}.")
            
            # Now we will rename this vertex so that it also contains its shortest path distance from the source vertex
            new_vertex_name = this_vertex_name + f" ({shortest_paths[this_vertex_name]})"
            
            # Update the mappings so we can keep track of the new and old vertex names
            org_to_new.update({ this_vertex_name : new_vertex_name })
            new_to_org.update({ new_vertex_name : this_vertex_name })
            
            # Perform the renaming
            this_vertex.rename(new_vertex_name)
            
            # Mark it as green to show it's been fully explored
            this_vertex.set_colour(finish_colours[1])
            GA.clear_text()
            GA.add_text(algorithm_text())
            GA.save_state()
            
        G.annotate(f"We have now fully explored this connected component of {G.name}.")
        GA.clear_text()
        GA.add_text(algorithm_text())
        GA.save_state()

    # Initially begin our BFS at the start vertex
    component_bfs(start_vertex_name, 0)

    # If persistent then we keep exploring all connected components of the graph
    if persistent:

        # Keep doing this for every connected component until the whole graph is explored
        while len(explored) != len(G.V):
            
            # Pick a new arbitrary start vertex
            new_start_vertex_name = list(set(original_vertex_names) - explored)[0]
            
            # We explore a new connected component, set it to be explored
            connected_components.append([new_start_vertex_name])
            explored.add(new_start_vertex_name)
            explored_list.append(new_start_vertex_name)
            
            # Any connected component away from the CC of the start vertex will have an infinite distance from the start
            shortest_paths[new_start_vertex_name] = float("inf")
            
            # Update all the visuals to accommodate for the new vertex discovery
            G.highlight_vertex(new_start_vertex_name, highlight_colour)
            G.annotate(f"Restart the search at vertex {new_start_vertex_name}.")
            GA.clear_text()
            GA.add_text(algorithm_text())
            GA.save_state()
            
            component_bfs(new_start_vertex_name, component_index= len(connected_components)-1)
            
        
        G.annotate(f"We have now fully explored {G.name} using Breadth-First-Search (BFS).")
    
    else:
        G.annotate(f"We have now fully explored all vertices reachable from {start_vertex_name} using Breadth-First-Search (BFS).")    
    
    
    # Rename all the renamed vertices back to their original names
    G.rename_vertices(new_to_org)
    
    GA.clear_text()
    GA.add_text(algorithm_text())
    GA.save_state()
    
    # Get rid of all the text we added before
    G.clear_highlighting()
    G.clear_annotations()
    GA.clear_text()
    
    # Restore the original colours of the graph's vertices
    G.assign_vertex_colours(original_colours)
    
    # Sort the shortest paths by their distance from the start vertex 
    shortest_paths = sorted(list(shortest_paths.items()), key = lambda x : x[1])
    
    # Return the explored vertices in order, the shortest paths from the start vertex, and the connected components
    return (explored_list, (start_vertex_name, shortest_paths), connected_components)