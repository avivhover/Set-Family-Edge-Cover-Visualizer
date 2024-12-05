from typing import List, Dict

from full_algorithm_solver import MyScene as Scene, Vertex, Edge
from set_family_edge_cover_vis.primal_dual_runner import Family

N = 7


class MyScene(Scene):
    def get_title(self) -> str:
        return "3-Constrained Forest"

    def get_vertices(self) -> List[Vertex]:
        return list(range(N))

    def get_edges(self) -> List[Edge]:
        return [(i, (i + 1) % N) for i in self.vertices]

    def get_costs(self) -> Dict[Edge, float]:
        return {e: (e[0] + e[1]) % (N + 2) + 1 for e in self.edges}

    def get_family(self) -> Family:
        return frozenset({frozenset((i,)) for i in self.vertices} |
                         {frozenset((i, j)) for index, i in enumerate(self.vertices) for j in self.vertices})

    def should_reverse_delete(self) -> bool:
        return False
