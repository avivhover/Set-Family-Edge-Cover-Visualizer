# Set Family Edge Cover Visualizer
This repository contains a visualizing tool for the primal-dual algorithm for solving the Set Family Edge Cover problem.

This project was created as part of a seminar in algorithms in the Open University of Israel, during semester 2024c.
It is based on the paper "Extending the primal-dual 2-approximation 
algorithm beyond uncrossable set families" by Zeev Nutov https://arxiv.org/abs/2307.08270.

## Demo visualizations
Two demo visualizations are provided. One is for solving the 3-Constrained Forest problem, and one for the Shortest Path
Problem. They are provided with both the final created video, and the code used to create them.

## How to create visualizations
The file `primal_dual_runner.py` contains a superclass called `Scene`. To create a new visualization, crete a new file
and inherit this class. Arguments of the graph, its costs, the set family, and other visualization parameters are passed
using abstract methods, so initialize those that are needed and override others depending on the desired arguments.
- get_vertices: The graph's vertices. Type: List of int
- get_edges: The graph's edges. Type: List of tuples, each tuple has 2 ints
- get_costs: The graph's edge costs. Type: Dict from edge (tuple of 2 ints) to int
- get_family: The set family to cover. Type: Frozenset of frozenset of vertices (ints)
- get_title (optional): The title to be shown at the start. Type: String, supports LaTeX
- should_reverse_delete (optional): Whether to perform the reverse-delete phase. Type: boolean. Default: True
- get_increment_delta (optional): The step size of the gradual increase in phase 1. Type: float. Default: 0.1
- get_num_highlighted_sample_sets (optional): The number of sets in the residual family to highlight in each iteration.
0 means that no sets are shown. None means to show all sets. Type: int or None. Default: 3

To finally run the visualization, run the command `python -m manim <YOUR-FILE-NAME>.py`. For Manim-specific parameters,
such as scene dimensions, you should check out [Manim's documentation](https://www.manim.community/).

Make sure you have Manim installed (version found in `requirements.txt`).