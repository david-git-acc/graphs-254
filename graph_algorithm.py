import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from graph_building_methods import create_graph
from helper_functions import paragraphise
from algorithms.bfs import *
from algorithms.dfs import *
from algorithms.dag import *
from algorithms.bipartiteness import *
from algorithms.scc import *
from algorithms.bcc import *
from algorithms.fordfulk import *
from algorithm_video_maker import make_video
import os, shutil


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
# when you want to run any graph defined algorithm from the algorithms folder, call the self.run_algorithm(alg,graph) method
class Graph_algorithm():
    
    # The frame folder states where all the frames/videos should go
    def __init__(self, capture : bool = True, frame_folder : str = "captures"):
            
        # These folders should never be used as folders to store frames
        illegal_folders = set([".","","algorithms"])
        
        if frame_folder in illegal_folders: raise Exception(f"Illegal folder name {frame_folder}")
            
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
        self.capturing : bool = capture
        
        # These are all the methods which will cause a new image to be created - any others will not
        self.saveable_methods : set[str] = set(["set_vertex_colour","add_edge","add_vertex","remove_edge","remove_vertex",
                                               "assign_vertex_colours","assign_edge_colours"])
        
        # Where the text containing the graph information stored will be placed relative to the axes
        self.text_pos = (0,1.05)
        
        # Stores axes text objects that contain information about the state of the graph on the photos
        self.recorded_info = []
        
        # We need to make sure the testing folder actually exists first
        if frame_folder != "" and self.capturing: 
            
            # Now make the directory again
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
        
        # We need to have a graph to save the state of
        if self.current_graph is None: raise Exception("empty graph algorithm object - add a graph using add_graph")
        
        # We will use these to determine what to call each frame
        this_graphs_name : str = self.current_graph.name
        graph_frame_count : int = self.frame_counter[this_graphs_name]
        
        # Save the figure with a filename based on the frame counter
        filename = f"f_{graph_frame_count}.png"
        
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
        
        # If there is no current graph (empty) then we have to raise an exception
        if self.current_graph is None: raise Exception("empty graph algorithm object - add a graph using add_graph")
        
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
        
    # Given an algorithm A and graph G, compute A(G).
    # algorithm: the algorithm to run
    # graph: the graph to run the algorithm on
    # capture : if set to true, images and a video will be produced. Otherwise, neither will be produced.
    # kill_existing : if set to true, deletes the existing contents of the folder
    # save_video : if set to true, we will also produce a video, otherwise only images
    # args and kwargs - you can specify the specific arguments to the algorithm, if it accepts any
    def run_algorithm(self, algorithm, *args, graph : Graph = None, capture : bool = True,
                      kill_existing : bool = True, save_video : bool = True,
                      fps : int = 1, seconds_per_image : float = 2.5, **kwargs) -> None:
        
        # Store the original capturing in a variable - we'll override the 
        # object's capturing preference for now but set it back after we finish
        remember = self.capturing
        self.capturing = capture
        
        # If no graph is specified, run on the current graph
        if graph is None: graph = self.get_current_graph()
        
        # Switch to the graph that you want to use for the algorithm
        self.switch_to_graph(graph)
        
        # The files will be located here
        filepath = self.frame_folder + "/" + graph.name
        
        # If we are capturing, delete the existing files to avoid issues with the video
        if capture and kill_existing: shutil.rmtree(filepath, ignore_errors=True)
        
        # Call the algorithm and run it, obtaining some result
        algorithm_result = algorithm(self, *args, **kwargs)

        if capture and save_video: make_video(graph.name + "_" + algorithm.__name__ + ".mp4", filepath,
                                              fps=fps,seconds_per_image=seconds_per_image)
        
        # Don't forget to set the capturing flag back to the original        
        self.capturing = remember
        
        return algorithm_result
        


s = ( 
'''
       1         4

s      2         5      t

       3         6       
''' )



edges = '''(s,1),(s,2),(s,3),(1,4),(1,5),(1,2),(2,5),(2,3),(3,6),(4,t),(4,5),(5,t),(5,6),(6,t),(6,2)'''

# Example weights
weights =[10,5,15,9,15,4,8,4,16,10,15,10,15,10,6]#list((np.random.rand(17)*100).astype(int))

# Build the graph according to the vertex schematic, edges and weights
G = create_graph(s, edges, weights, vertexcolour="orange")




flows = {edge_name : 0 for edge_name in G.edges()}
capacities = { edge_name : G.get_edge(*edge_name).weight for edge_name in G.edges()}


# Instantiate the graph algorithm on the graph
GA = Graph_algorithm()

print(GA.run_algorithm(ford_fulkerson, graph=G, source_name="s",target_name="t"))




# print( GA.run_algorithm(scc_algorithm, graph=G, seconds_per_image=1, capture=True) )






