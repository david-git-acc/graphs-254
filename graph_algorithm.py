import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from graph_building_methods import create_graph
from helper_functions import paragraphise
from algorithms.bipartiteness import test_for_bipartiteness
import os


# This class will be a wrapper around the Graph class and will be used for building graph algorithm visualisations
# It takes in a set of graphs, which are a dictionary (name - object), and allows you to switch between graphs
# At any given time the graph algorithm will have a "current graph" which is the graph that will be operated on
# if an algorithm is called on the graph algorithm object. 
# 
# When you want to call any graph function, you do so through the graph_algorithm object as a wrapper, passing in
# the reference to the graph you want to apply this graph function to. 
# e.g for a graph G and graph algorithm GA containing G, then: 
# G.add_edge("A","B") becomes GA.add_edge(G,"A","B") 
# it will always follow this format for any graph function - if it previously had n arguments it will now have n+1 arguments
# 
# when you want to run any graph defined algorithm from the algorithms folder, just run algorithm_name(GA)
class Graph_algorithm():
    
    def __init__(self, frame_folder : str = "frames"):
            
        # Each graph algorithm will store a number of graphs, referenced by their name
        # Of course it has to contain at least one graph otherwise this class is useless
        self.graphs : dict[str,Graph] = {}
        
        # At any given time the graph algorithm will be "looking" at a particular graph, so we need to store this information
        self.current_graph = None
        
        # This counter uniquely identifies individual frames for each graph
        self.frame_counter : dict[str, int] = {}
        
        # Where the frames will be stored
        self.frame_folder = frame_folder
        
        # Whether or not we will save images and create a video
        self.capturing : bool = True
        
        # These are all the methods which will cause a new image to be created - any others will not
        self.saveable_methods : set[str] = set(["set_vertex_colour","add_edge","add_vertex","remove_edge","remove_vertex",
                                               "assign_vertex_colours","assign_edge_colours"])
        
        # Where the text containing the graph information stored will be placed relative to the axes
        self.text_pos = (0,1.025)
        
        # Stores axes text objects that contain information about the state of the graph on the photos
        self.recorded_info = []
        
        # We need to make sure the testing folder actually exists first
        if frame_folder != "": 
            
            os.makedirs(frame_folder, exist_ok=True)
        
        
        # Dynamically wrap each method of the Graph_algorithm class
        for name in dir(Graph):
            method = getattr(Graph, name)
            
            # If the method is a method (and not an attribute or class) then we will replicate it
            if callable(method) and not name.startswith("__"):
                
                # We will create a wrapped method and set it to be a method for graph algorithm
                wrapped_method = self.wrap(method)
                
                # Now its method becomes ours
                setattr(self, name, wrapped_method)
        
        
    # The wrapper method used - we will use this to do extra things every time we make a graph operation
    def wrap(self, method):
        def wrapped_method(graph: Graph, *args, **kwargs):
            
            # Obtain the method that specifically belongs to the graph we want
            this_graphs_method = getattr(graph, method.__name__)

            # Obtain the result from computing the method - this is the original result from the graph
            result = this_graphs_method(*args, **kwargs)
           
            # Now we can perform a save or whatever else we want afterwards
            if method.__name__ in self.saveable_methods:
                # Now we save the current state of the graph so we can use it for the animation
                self.save_state()
            
            # Return the result of the original method
            return result
        
        return wrapped_method
        
    # Add a graph to the graph algorithm object
    def add_graph(self, graph : Graph) -> None:
        
        # We will need to track the graphs and how many frames they've elapsed
        self.graphs.update({ graph.name : graph })
        self.frame_counter.update({ graph.name : 0})
        
        if self.current_graph is None:
            # If we had no graphs before then we must switch to this one
            self.switch_to_graph(graph)
        else:
            # Switch back to the current one, in case the figure was changed
            self.switch_to_graph(self.current_graph)
  
    
    # Save the current state of the current graph as a frame
    def save_state(self) -> None:
        
        # We will use these to determine what to call each frame
        this_graphs_name : str = self.current_graph.name
        graph_frame_count : int = self.frame_counter[this_graphs_name]
        
        # Save the figure with a filename based on the frame counter and the specific graph it's based on
        filename = f"{this_graphs_name}_f{graph_frame_count}.png"
        
        if self.capturing:
            
            # We will store the frames in a subfolder specifically meant for this graph
            subfolder : str = self.frame_folder + "/" + this_graphs_name
            
            # Make the directory if it doesn't already exist
            os.makedirs(subfolder, exist_ok=True)
            
            plt.savefig(subfolder + "/" + filename)
            
            # Increment the number of frames used for the given graph
            self.frame_counter[self.current_graph.name] += 1
    
    # Get the current graph being examined by the algorithm
    def get_current_graph(self) -> Graph:
        self.switch_to_graph(self.current_graph)
        return self.current_graph
        
    # Switch the visualisation view to a particular graph (we can only view one graph (and hence one figure) at a time)
    # If you switch to a graph not in the set of graphs, it will automatically add it to the set
    def switch_to_graph(self, graph : Graph) -> None:
        
        # If not already in the list of graphs, add it
        if self.graphs.get(graph.name) is None: self.add_graph(graph)
        
        # Switch the current graph over so we know which one to modify
        self.current_graph = graph

        # Move the current figure to the figure of this graph
        plt.figure(graph.fig.number)
    
    # Get an arbitrary graph from the graph algorithm
    def get_graph(self) -> Graph: return self.graphs[self.graph_names()[0]]
                
    # Get all the names of the graph that this graph algorithm stores
    def graph_names(self) -> list[str]: return list(self.graphs.keys())
    
    # Get rid of all the text on the screen
    def clear_text(self) -> None:
        for text in self.recorded_info:
            try: text.remove()
            except: pass
    
        self.recorded_info = []
    
    
    # Add text to the top of the current graph, symmetrically opposite the annotation text
    # This is usually used to store information about the graph
    def add_text(self, text : str) -> None:
        
        # Prevent the text from going off the edge of the figure by adding newlines, making it a paragraph
        paragraphised_text : str = paragraphise(text, self.current_graph.characters_per_line)
        
        # Create the text object - also add a box around it to make it more pronounced
        info = self.current_graph.ax.text(*self.text_pos, paragraphised_text, fontsize = self.current_graph.legendsize, 
                              ha="left",va="center", 
                              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
        self.recorded_info.append(info)
        
  
    
s = ( 
'''
    A       D
                G
    B               I
                H
    C       F
    

''' )



edges = "(B,A),(B,C),(C,F),(A,D),(D,G),(F,H),(H,F) , (G,D), (G,I),(H,I),(B,I), (B,D),(D,B) , (B,F), (F,B), {H,F}"

# Example weights
weights = []

# Build the graph according to the vertex schematic, edges and weights
G = create_graph(s, edges, weights, vertexcolour="lime")
H = create_graph(s, edges, weights, vertexcolour="brown", name="H")

H.remove_edge("C","F")
H.remove_edge("A","D")


# Instantiate the graph algorithm on the graph
GA = Graph_algorithm( "frame_folder")

GA.add_graph(G)
GA.add_graph(H)

print(test_for_bipartiteness(GA))

GA.switch_to_graph(H)

print(test_for_bipartiteness(GA))
