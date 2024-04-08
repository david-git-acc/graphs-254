from graph import Graph
import matplotlib.pyplot as plt

def verify_graph_for_fulkerson(graph : Graph, source_vertex : str, target_vertex : str) -> None:
    
    """
    Determine if the given graph, with designated source and target vertices, fits the constraints for a flow network.

    Raises:
        Exception: Graph is not strictly directed - if edge (A,B) exists then edge (B,A) cannot
        Exception: No or invalid assignment for the capacity of an edge
        Exception: Source vertex has incoming edges, a contradiction
        Exception: Target vertex has outgoing edges, a contradiction

    """
    
    # This code is fairly self explanatory from the exception messages
    for source_name, dest_name in graph.edges():    
        if graph.get_edge(dest_name, source_name) is not None: 
            raise Exception(f"Flow graph must be strictly directed - edges {source_name}-{dest_name} and {dest_name}-{source_name} both exist in {graph.name}")
            
        edge = graph.get_edge(source_name, dest_name)
        
        if edge.weight is None:
            raise Exception(f"Edge {source_name}-{dest_name} has no assigned capacity")
        
        
        try:
            if int(edge.weight) < 0:
                raise Exception(f"catch me")
        except:
            raise Exception(f"Capacity for edge {source_name}-{dest_name} must be a positive integer, not {edge.weight}")
    
    if len(graph.get_vertex(source_vertex).incoming_edges) != 0:
        raise Exception(f"Source vertex {source_vertex} has incoming edges")
    
    if len(graph.get_vertex(target_vertex).outgoing_edges) != 0:
        raise Exception(f"Target vertex {target_vertex} has outgoing edges")


def remove_all_after_list(listing : list, element):
    for i in range(len(listing)):
        if listing[i] == element: 
            break

    return listing[0:i]


# Get the flow of a flow network  the sum of all incoming flows (no outgoing flows are permitted)
def get_flow(graph : Graph, target_vertex_name : str, flows : dict[tuple[str,str]]) -> int:
    
    # Sum of flows for each incoming edge
    return sum([ flows[(vertex.name, target_vertex_name)] 
         for vertex in  graph.get_vertex(target_vertex_name).incoming_edges.values()])

    
# Find an augmenting path in the graph, returning the path and its flow size (or None if no path exists)
def find_augmenting_path(graph : Graph, source_vertex : str, target_vertex : str, 
                         f : dict[str,int],  highlight_colour : str = "gold") -> None:
    
    # Avoid stores all vertices to avoid trying to connect a path to
    avoid = set([])
    path = []
    pathv = set([source_vertex])
    last_vertex_name = source_vertex
    
    # Keep searching until we find the target or we exhaust all paths
    while last_vertex_name != target_vertex and len(pathv) != 0:
        
        last_vertex = graph.get_vertex(last_vertex_name)
        neighbours = last_vertex.connected_vertices()
        
        # We will use this variable to check if a path has been identified or not, and hence whether to backtrack
        path_edge_found = False
        for neighbour in neighbours:
            
            # Avoid rechecking the same vertex twice - and don't form cycles
            if neighbour.name not in avoid and neighbour.name not in pathv:
                
                # Get the edge to check if it would work as a new path edge
                edge_name = (last_vertex_name, neighbour.name)
    
                # The remaining capacity of an edge flow is the flow of the reverse edge
                remaining_capacity = f[edge_name[::-1]]
                
                # If it's 0 then we can't have an augmenting path this way
                if remaining_capacity != 0:
                    path_edge_found=True
                    path.append(edge_name)
                    pathv.add(neighbour.name)
                    
                    # Get the last vertex in the path and its neighbours so we can find a new path
                    last_vertex_name = neighbour.name
                    break
                
        # If we can't find any path from this vertex, add to avoid so we don't explore it again
        if not path_edge_found:
            # Remove from the path and add to set of vertices to avoid exploring
            pathv.remove(last_vertex_name)
            avoid.add(last_vertex_name)

            # Remove the last edge from the path 
            if len(path) != 0: path.pop()

            # The new last vertex name is the last vertex in the path
            last_vertex_name = path[-1][1] if path else source_vertex
            
            
    
    # If no remaining path exists, we return None to express this            
    if len(path) == 0: return (None,0)
    
    # The size of the augmenting path is the minimum flow along it
    # logically must be >0 or else we wouldn't have explored this path
    pathflow = min([  f[edge_name[::-1]] for edge_name in path ])
    
    # Highlight the edges so we can identify the augmenting path
    for v in pathv: graph.highlight_vertex(v, highlight_colour)
    for e in path: graph.highlight_edge(e, highlight_colour)

    # Give back the path flow as a result
    return (path, pathflow)


        
   
         

# Add all the "backflow" edges to the graph - the capacity of the original edge is the flow of the new edge
def add_reverse_flows(graph : Graph, capacities) -> None:
    for source_name, dest_name in graph.edges():
        graph.add_edge(dest_name, source_name, False, capacities[source_name, dest_name], graph.edgecolour, "dotted")
        
        
def ford_fulkerson(GA, source_name : str, target_name : str, 
                   capacities : dict[tuple[str,int]] = None, fullcolour: str = "red") -> None:
    
    
    G : Graph = GA.get_current_graph()
    
    # Check that the graph is actually suitable for performing the Fulkerson algorithm on
    verify_graph_for_fulkerson(G, source_name, target_name)
    
    # Instantiate the flows and capacities of each edge in the graph
    if not capacities: capacities = { edge_name : G.get_edge(*edge_name).weight for edge_name in G.edges() }
    
    
    # Store the original edges of the flow network so we can remove the added ones afterwards
    original_edges = set(G.edges())
    
    # Get the original weights of the edges so we can set them back
    original_edgeweights = { edge_name : G.get_edge(*edge_name).weight for edge_name in G.edges()}
    
    # Get the original text colours of the edges so we can put them back later
    original_textcolours = G.get_edge_textcolours()

    
    # Initialise all flows at 0
    flows = { edge_name : 0 for edge_name in G.edges() }
    flows.update({ edge_name[::-1] : capacities[edge_name] for edge_name in G.edges() })
    
    # Set the capacities initially so we can get an initial look at the graph 
    for edge_name in G.edges(): G.get_edge(*edge_name).set_weight(f"0 / {capacities[edge_name]}")
    
    G.annotate(f"We will use the Ford-Fulkerson algorithm to find the maximum flow of {G.name}, with source vertex {source_name} and target vertex {target_name}. Initialise all flows, and the maximum flow as 0.")
    GA.save_state()

    # Now set the weights to 0 to show there is no flow in these edges yet
    for edge_name in G.edges(): G.get_edge(*edge_name).set_weight(0)

    # Add all reverse flows - as all normal flows start at 0, then 
    add_reverse_flows(G, capacities)
    
    
    # We start with an initial flow of 0
    flow = 0
    
    # This is the text that we show to the user
    def algorithm_text() -> str:
        
        # Get the flows of the non-residual edges in readable string format
        flows_str = { edge_name[0] + "-" + edge_name[1] : flows[edge_name] for edge_name in original_edges }
        
        # Stringify them and remove quotation marks for clarity
        flows_str = str(flows_str).replace("'","")
        
        return f"Current maximum flow: {flow}\nFlows: {flows_str}"
    
    G.annotate("Add the reverse flow edges to create the residual flow network.")
    GA.clear_text()
    GA.add_text(algorithm_text())
    GA.save_state()
    
    
    # Augment the flow of the graph according to an augmenting path and its flowsize
    def augment_flow(path : list[tuple[str,str]], flowsize : int) -> None:
        
        
        
        # For each edge in the augmenting path flow, we increase/decrease the flow
        for edge_name in path:
            
            # Name of the reverse edge so we can decrease its flow
            revname = edge_name[::-1]
            
            # Increase/decrease the flow in the corresponding directions            
            flows[edge_name] += flowsize
            G.get_edge(*edge_name).set_weight(flows[edge_name])
            
            flows[revname] -= flowsize
            G.get_edge(*revname).set_weight(flows[revname])
            
            edgetextcolour = fullcolour if flows[edge_name[::-1]] == 0 else G.edge_textcolour                
            G.get_edge(*edge_name).set_textcolour(edgetextcolour)
            


    
    # Keep searching for augmenting paths until we can't find any more
    augmenting_path, flowsize = find_augmenting_path(G, source_name, target_name, flows)
    while augmenting_path is not None:
        
        # Augment the flow so we can move onto the next step of the algorithm
        augment_flow(augmenting_path, flowsize)

        flow = get_flow(G, target_name, flows)
   
        # Convert the path into a readable format so we can print it out
        path_strformat = "-".join([x[0] for x in augmenting_path  ] +[ augmenting_path[-1][1]]).replace("'","")
        G.annotate(f"We have found the augmenting path {path_strformat} with an augmenting flow of {flowsize}. The maximum flow is now {flow}.")
        GA.clear_text()
        GA.add_text(algorithm_text())
        GA.save_state()
        
        # Remove the highlighting of the previous edge
        G.clear_highlighting()
        
        # Find another augmenting path to choose from
        augmenting_path, flowsize = find_augmenting_path(G, source_name, target_name, flows)
        
    
    G.annotate(f"There are no augmenting paths remaining. Our maximum flow for {G.name} is {flow}.")
    GA.clear_text()
    GA.add_text(algorithm_text())
    GA.save_state()
    
    # Remove all the residual edges and their quantities in the flow network
    for edge_name in G.edges():
        if edge_name not in original_edges:
            G.remove_edge(*edge_name)
            del flows[edge_name]
    
    
    for edge_name in original_edges:
        edge = G.get_edge(*edge_name)
        edge.set_weight(f"{edge.weight} / {capacities[(edge.source.name, edge.destination.name)]}")
    

    GA.save_state()

    # Reset the edge weights back to their original values    
    for edge_name in G.edges():
        G.get_edge(*edge_name).set_weight(original_edgeweights[edge_name])
        
    # Set the text colours back to their original values
    G.assign_edge_textcolours(original_textcolours)
    
    G.clear_highlighting()
    GA.clear_text()
    
    # Return the assignment of flows and the maximum flow size
    return (flows, flow)
    