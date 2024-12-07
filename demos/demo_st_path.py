from typing import List, Set, FrozenSet, Dict

from primal_dual_runner import MyScene as Scene, Vertex, Edge, powerset

s = 4
t = 3


class MyScene(Scene):
    def get_title(self) -> str:
        return "st-Path"

    def get_vertices(self) -> List[Vertex]:
        return list(range(8))

    def get_edges(self) -> List[Edge]:
        return [(0, 1), (1, 2), (2, 3),
                (0, 4), (1, 5), (2, 6), (3, 7),
                (4, 5), (5, 6), (6, 7)]

    def get_costs(self) -> Dict[Edge, float]:
        return {e: (e[0] + e[1]) % 10 + 1 for e in self.edges}

    def get_family(self) -> Set[FrozenSet[Vertex]]:
        return {S for S in powerset(self.vertices) if s in S and t not in S}

    def get_increment_delta(self) -> float:
        return 1
