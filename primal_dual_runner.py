import math
from abc import abstractmethod
from itertools import chain, combinations
from random import choice, sample
from typing import Tuple, FrozenSet, Iterable, Optional

from manim import *
from manim.typing import Point3D

# Type definitions
Vertex = int
VerSet = FrozenSet[Vertex]
Family = FrozenSet[FrozenSet[Vertex]]
Edge = Tuple[int, int]
EdgSet = Iterable[Edge]

NDIGS = 1   # Digits after decimal point to round to
DEFAULT_NUM_HIGHLIGHTED_SETS = 3
DEFAULT_TITLE = "Set Family Edge Cover"
DEFAULT_INCREMENT_DELTA = 0.1

RESIDUAL_FAMILY_TITLE = r"\mathcal{F}^{I}"
CORES_OF_RESIDUAL_TITLE = r"C(\mathcal{F}^{I})"
REVERSE_DELETE_TITLE = "Reverse Delete"


def powerset(s):
    """
    Given a set S, computes its powerset P(S).
    :param s: A set.
    :return: s's powerset.
    """
    return set(chain.from_iterable(map(frozenset, combinations(s, i)) for i in range(len(s) + 1)))


def crosses(edge: Edge, subset: VerSet) -> bool:
    """
    Given an edge and a set, determines whether the edge crosses the set.
    :param edge: An edge.
    :param subset: A nontrivial set of vertices.
    :return: True if the edge has exactly one end in the set and one end outside it. False otherwise.
    """
    return len(set(edge) & subset) == 1


def compute_residual(F: Family, I: EdgSet) -> Family:
    """
    Given a set family F and an edge set I, compute the residual family F^I of F with respect to I. The residual family
    is the subfamily of all sets in F that are not crossed by I.
    :param F: A set family.
    :param I: An edge set.
    :return: The subfamily of F of all sets not crossed by I.
    """
    return frozenset({S for S in F if not any(crosses(edge, S) for edge in I)})


def get_cores(F: Family) -> Family:
    """
    Given a set family, compute its cores C(F). Cores of a family are its inclusion-minimal members.
    :param F: A set family.
    :return: The family of F-cores F(C).
    """
    return frozenset({C for C in F if not any(C > S for S in F)})


def cover_all_sets(I: EdgSet, F: Family) -> bool:
    """
    Computes whether I covers F. In notation, whether d_I(S) >= 1 for all S in F.
    This can be seen as whether the residual family F^I is empty.
    :param I: An edge set.
    :param F: A set family.
    :return: True if I covers F, False otherwise.
    """
    for S in F:
        if not any(crosses(e, S) for e in I):
            return False
    return True


def next_to(obj: Mobject, scale: float = 0.2) -> Point3D:
    """
    Given a graphic object, gives coordinates to draw something near it.
    :param obj: An object in Euclidean space.
    :param scale: How far away the object should be.
    :return: A 3D point near the object, scale units away from the center.
    """
    if isinstance(obj, VGroup):
        center = choice(obj.submobjects).get_center()
        scale += 0.1 * len(obj.submobjects)
    else:
        center = obj.get_center()
    vect = (center - ORIGIN)
    return center + vect / np.linalg.norm(vect) * scale


class MyScene(Scene):
    def __init__(self):
        super().__init__()
        # Load parameters from inheriting classes
        self.vertices = self.get_vertices()
        self.edges = self.get_edges()
        self.costs = self.get_costs()
        self.family = self.get_family()
        self.reverse_delete = self.should_reverse_delete()
        self.increment_delta = self.get_increment_delta()
        self.num_highlighted_sample_sets = self.get_num_highlighted_sample_sets()

        # Create all the initial graphical representations
        self.g = Graph(
            vertices=self.vertices,
            edges=self.edges,
            layout_scale=2
        )
        self.label = MathTex(self.get_title()).shift(LEFT * 3 + UP * 3)
        self.edge_cost_labels = {
            e: DecimalNumber(self.costs[e], num_decimal_places=0, font_size=36).move_to(next_to(self.g.edges[e]))
            for e in self.edges
        }
        self.add(*self.edge_cost_labels.values())

        # Initialize the output edge set ([] simulates the empty set)
        self.I: List[Edge] = []

    @abstractmethod
    def get_vertices(self) -> List[Vertex]:
        """
        :return: The graph's vertices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_edges(self) -> List[Edge]:
        """
        :return: The graph's edges.
        """
        raise NotImplementedError

    @abstractmethod
    def get_costs(self) -> Dict[Edge, float]:
        """
        :return: a mapping of a cost for each edge.
        """
        raise NotImplementedError

    @abstractmethod
    def get_family(self) -> Family:
        """
        :return: The set family to cover.
        """
        raise NotImplementedError

    @abstractmethod
    def get_title(self) -> str:
        """
        :return: The title of the scene.
        """
        return DEFAULT_TITLE

    def should_reverse_delete(self) -> bool:
        """
        :return: Whether to perform the reverse-delete phase.
        """
        return True

    def get_increment_delta(self) -> float:
        """
        :return: The increment size in each step in the first phase.
        """
        return DEFAULT_INCREMENT_DELTA

    def get_num_highlighted_sample_sets(self) -> Optional[int]:
        """
        :return: How many sample sets should be highlighted in each iteration.
        0 means no sets are shown. None means all sets are shown.
        """
        return DEFAULT_NUM_HIGHLIGHTED_SETS

    def get_ellipse(self, v_set: VerSet) -> VMobject:
        """
        Given a set of vertices, finds a 2D ellipse that circles them.
        If |v_set| = 1, then a circle is shown.
        If |v_set| = 2, then an ellipse encompasses both vertices. Assumes not 3 vertices are on the same line.
        If |v_set| > 2, then each vertex in circled in its own circle. This is to avoid confusion with other vertices
        in the scene. The circles are uniquely colored from other calls to this function.
        :param v_set: A set of vertices to circle (not their 2D objects).
        :return: A 2D object to drw that encompasses the given vertices.
        """
        if len(v_set) == 1:
            return Circle(radius=0.5, color=GRAY).move_to(self.g.vertices[next(iter(v_set))])
        elif len(v_set) == 2:
            v1, v2 = list(v_set)
            v1, v2 = self.g.vertices[v1], self.g.vertices[v2]
            v1 = v1.get_x(), v1.get_y()
            v2 = v2.get_x(), v2.get_y()
            center = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2
            vect = (v1[0] - v2[0]), (v1[1] - v2[1])
            dist = (vect[0] ** 2 + vect[1] ** 2) ** 0.5
            angle = math.atan(vect[1] / vect[0])
            return Ellipse(width=dist * 1.5, height=1, color=GRAY).move_to(np.array((*center, 0))).rotate(angle)
        else:
            group_color = random_color()
            circles = []
            for v in v_set:
                circles.append(Circle(radius=0.5, color=group_color).move_to(self.g.vertices[v]))
            return VGroup(*circles)

    def change_label(self, new_txt: str) -> None:
        """
        Replaces the scene's title.
        :param new_txt: The new text to be presented. Supports LaTeX.
        :return: None.
        """
        new_label = MathTex(new_txt).move_to(self.label)
        self.play(ReplacementTransform(self.label, new_label))
        self.label = new_label

    def highlight_sets(self, sets: List[VerSet]) -> None:
        """
        Briefly encircles a sets of vertices in order to draw attention to them.
        If the sets are small (<=2 elements) then they ar encircled simultaneously. Otherwise, they are encircled
        successively.
        :param sets: The set of vertices to encircle.
        :return: None.
        """
        self.change_label(RESIDUAL_FAMILY_TITLE)
        ellipses = [self.get_ellipse(S) for S in sets]
        if all(len(S) <= 2 for S in sets):
            # Show simultaneously
            self.play((Create(el) for el in ellipses))
            self.wait()
            self.play((Uncreate(el) for el in ellipses))
        else:
            # Show successively
            for ellipse in ellipses:
                self.play(Create(ellipse, lag_ratio=0))
                self.play(Uncreate(ellipse, lag_ratio=0))

    def draw_cores(self, cores: Family) -> Dict[VerSet, VMobject]:
        """
        Encircles all the cores of F.
        This shows all the cores simultaneously, but as long as the family is Uncrossable, Semi-Uncrossable or Pliable,
        the cores should be disjoint and have no ambiguity.
        :param cores: A vertex set family.
        :return: A dictionary of all the objects drawn. The dict maps each core to its visual encircling component.
        """
        self.change_label(CORES_OF_RESIDUAL_TITLE)
        ellipses = {}
        for c in cores:
            ellipses[c] = (self.get_ellipse(c))
        self.play(
            (Create(ellipse, lag_ratio=0) for ellipse in ellipses.values())
        )
        return ellipses

    def increment_cores(self, cores: Family, values: Dict[VerSet, float], ellipses: Dict[VerSet, VMobject]) -> Edge:
        """
        Gradually increases the dual variables corresponding to the cores. This is the main part of each iteration in
        phase 1 of the algorithm.
        :param cores: The family of F-cores.
        :param values: A dictionary of all dual variables (y_S). Works on this dict mutably to store new values for
        later iterations.
        :param ellipses: The ellipses that encircle each core. Shows the values of the variables next to each one.
        :return: The tight edge that was achieved.
        """
        labels = {
            c: DecimalNumber(values[c], num_decimal_places=NDIGS).move_to(next_to(ellipses[c], 1)) for c in cores}
        self.play((Create(l) for l in labels.values()))
        while True:
            # Look for a tight edge
            for e in self.edges:
                if e in self.I:
                    continue
                crossed = [S for S in self.family if crosses(e, S)]
                # If edge e is tight
                if sum(values[S] for S in crossed) >= self.costs[e]:
                    self.play(Wiggle(self.edge_cost_labels[e], scale_value=2))
                    # Show sets that have values from previous iterations, but are not currently shown.
                    undrawns = []
                    for undrawn_c in crossed:
                        if undrawn_c in cores:
                            continue
                        if values[undrawn_c]:
                            ellipse = self.get_ellipse(undrawn_c)
                            undrawns.append(ellipse)
                            undrawns.append(
                                DecimalNumber(values[undrawn_c],num_decimal_places=NDIGS
                                              ).move_to(next_to(ellipse, 0.5)))
                    if undrawns:
                        self.play(Create(o, lag_ratio=0) for o in undrawns)
                    self.play(FadeToColor(self.g.edges[e], color=RED))
                    if undrawns:
                        self.play(Uncreate(o, lag_ratio=0) for o in undrawns)
                    self.play((Uncreate(l) for l in labels.values()))
                    self.play((Uncreate(e, lag_ratio=0) for e in ellipses.values()))
                    return e
            # Increase variables uniformly
            for c in cores:
                values[c] = round(values[c] + self.increment_delta, NDIGS)
            self.play((labels[c].animate.set_value(values[c]) for c in cores), run_time=1)

    def sample_sets(self, family: Family) -> List[VerSet]:
        """
        Samples a random selection of sets from the family to be highlighted.
        :param family: A set family.
        :return: The sets selected.
        """
        if self.num_highlighted_sample_sets is None:
            return list(family)
        return sample(family, min(DEFAULT_NUM_HIGHLIGHTED_SETS, len(family)))

    def construct(self):
        """
        The `main` function of the animation.
        :return: None.
        """
        self.add(self.g, self.label)

        residual = compute_residual(self.family, self.I)
        cores = get_cores(residual)
        y = {S: 0 for S in residual}
        self.wait(2)
        # Phase 1
        while cores:
            selected_sets = self.sample_sets(residual)
            self.highlight_sets(selected_sets)
            ellipses = self.draw_cores(cores)
            added_edge = self.increment_cores(cores, values=y, ellipses=ellipses)
            self.I.append(added_edge)
            residual = compute_residual(self.family, self.I)
            cores = get_cores(residual)

        # Phase 2
        if self.reverse_delete:
            self.change_label(REVERSE_DELETE_TITLE)
            self.wait()
            for e in reversed(self.I):
                if cover_all_sets(set(self.I) - {e}, self.family):
                    self.I.remove(e)
                    self.play(Wiggle(self.edge_cost_labels[e], scale_value=2))
                    self.play(self.g.edges[e].animate.set_stroke(color=WHITE))
