from abc import ABC, abstractmethod
from bisect import insort
from collections import deque

import networkx as nx

__all__ = ["UndirectedGraph"]


class AbstractGraph(ABC):
    """Abstract base class for graph structures."""

    def __init__(self, adjacency_list=None) -> None:
        self.adjacency_list: list[list[int]] = adjacency_list or []

    @abstractmethod
    def add_edge(self, vertex1: int, vertex2: int) -> None:
        pass

    def add_vertex(self) -> None:
        """Adds a new vertex to the graph."""
        self.adjacency_list.append([])

    @property
    def vertices(self):
        """List of vertices in the graph.

        Returns:
            list[int]: A list of vertex indices.
        """
        return list(range(len(self.adjacency_list)))

    @property
    def nvertices(self) -> int:
        """The number of vertices in the graph."""
        return len(self.vertices)

    @property
    @abstractmethod
    def edges(self):
        pass

    @property
    @abstractmethod
    def nedges(self) -> int:
        pass

    def __repr__(self):
        return f"{type(self).__name__}({self.adjacency_list})"

    def dfs(self, start_vertex: int) -> set[int]:
        """Performs depth-first search starting from a given vertex.

        Traverses the graph using depth-first search and returns the set of visited vertices.

        Args:
            start_vertex (int): The starting vertex for DFS.

        Returns:
            set[int]: A set of vertices visited during the DFS.
        """
        visited: set[int] = set()
        stack: list[int] = [start_vertex]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                stack.extend(reversed(self.adjacency_list[vertex]))

        return visited

    def bfs(self, start_vertex: int) -> set[int]:
        """Performs breadth-first search starting from a given vertex.

        Traverses the graph using breadth-first search and returns the set of visited vertices.

        Args:
            start_vertex (int): The starting vertex for BFS.

        Returns:
            set[int]: A set of vertices visited during the BFS.
        """
        visited: set[int] = set()
        queue: deque[int] = deque([start_vertex])

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(self.adjacency_list[vertex])  # No need to reverse

        return visited

    @property
    def connected_components(self):
        """Identifies and returns all connected components of the graph.

        Each connected component is represented as a set of vertices.

        Returns:
            list[set[int]]: A list of sets, each set containing the vertices of a connected component.
        """
        visited: set[int] = set()
        components: set[frozenset[int]] = set()
        for vertex in self.vertices:
            if vertex not in visited:
                component: set[int] = self.dfs(vertex)
                components.add(frozenset(component))
                visited.update(component)
        return components


class UndirectedGraph(AbstractGraph):
    """Represents an undirected graph.

    Extends `AbstractGraph` with specific implementations for undirected graphs.
    """

    def __init__(self, adjacency_list=None) -> None:
        super().__init__(adjacency_list)
        self._nedges = type(self).count_edges(self.adjacency_list)

    @classmethod
    def from_adjacency_matrix(cls, matrix):
        """Creates an instance from an adjacency matrix.

        Args:
            matrix (list[list[int]]): A square matrix representing graph connections.

        Returns:
            UndirectedGraph: A new instance.
        """
        graph = cls()
        n = len(matrix)
        for _ in range(n):
            graph.add_vertex()
        for i in range(n):
            for j in range(i, n):  # Start from i to avoid duplicates
                if matrix[i][j] != 0:
                    graph.add_edge(i, j)
        return graph

    def to_adjacency_matrix(self):
        """Converts the graph to an adjacency matrix representation.

        Returns:
            list[list[int]]: The adjacency matrix of the graph.
        """
        matrix = [[0 for _ in range(self.nvertices)] for _ in range(self.nvertices)]
        for vertex in range(self.nvertices):
            for adjacent in self.adjacency_list[vertex]:
                matrix[vertex][adjacent] = 1
                matrix[adjacent][vertex] = 1  # Since the graph is undirected
        return matrix

    @classmethod
    def from_edges(
        cls, edges: list[tuple[int, int]] | set[tuple[int, int]], nvertices: int
    ):
        """Creates an instance from a list of edges and a specified number of vertices.

        Args:
            edges (list[tuple[int, int]] | set[tuple[int, int]]): The edges.
            nvertices (int): The total number of vertices in the graph.

        Returns:
            UndirectedGraph: A new instance.
        """
        graph = cls()
        for _ in range(nvertices):  # We need to know all vertices in advance
            graph.add_vertex()  # If not, what if vertices with high indices do not have edges?
        for edge in edges:
            vertex1, vertex2 = edge
            # Ensure that both vertices exist in the graph
            if vertex1 >= nvertices or vertex2 >= nvertices:
                raise IndexError("vertex index out of range")
            # Add the edge to the graph, ensuring symmetry
            if vertex2 not in graph.adjacency_list[vertex1]:
                insort(graph.adjacency_list[vertex1], vertex2)
            if vertex1 != vertex2 and vertex1 not in graph.adjacency_list[vertex2]:
                insort(graph.adjacency_list[vertex2], vertex1)
        graph._nedges = cls.count_edges(graph.adjacency_list)
        return graph

    @property
    def edges(self):
        """Set of edges in the graph.

        For an undirected graph, each edge is represented as a tuple `(v1, v2)`.

        Returns:
            set[tuple[int, int]]: A set of tuples, each tuple representing an edge.
        """
        all_edges = set()
        for vertex in range(len(self.adjacency_list)):
            for adj_vertex in self.adjacency_list[vertex]:
                if vertex <= adj_vertex:  # To avoid duplicates in an undirected graph
                    all_edges.add((vertex, adj_vertex))
        return all_edges

    @property
    def nedges(self):
        """The number of edges in the graph."""
        return self._nedges

    @staticmethod
    def count_edges(adjacency_list):
        nedges = 0
        for vertex, adj_list in enumerate(adjacency_list):
            for adj_vertex in adj_list:
                if adj_vertex >= len(adjacency_list):
                    raise IndexError("Adjacency list is invalid!")
                if vertex not in adjacency_list[adj_vertex]:  # Check symmetry
                    raise ValueError(f"Missing edge {adj_vertex} => {vertex}!")
                if vertex == adj_vertex:  # Count self-loops twice
                    nedges += 2
                else:  # Count other edges once
                    nedges += 1
        nedges //= 2  # Correcting the count by dividing by 2
        return nedges

    def add_edge(self, vertex1: int, vertex2: int) -> None:
        """Adds an undirected edge between two vertices in the graph.

        Ensures that the graph remains undirected by adding an edge symmetrically to both
        vertices' adjacency lists. Raises an `IndexError` if either vertex does not exist.
        Raises a `ValueError` if the edge already exists.

        Args:
            vertex1 (int): The index of the first vertex.
            vertex2 (int): The index of the second vertex.

        Raises:
            IndexError: If one or both vertices are not found in the graph.
            ValueError: If an edge between the given vertices already exists.
        """
        if vertex1 >= len(self.adjacency_list) or vertex2 >= len(self.adjacency_list):
            raise IndexError("one or both vertices not found in graph.")

        if vertex2 in self.adjacency_list[vertex1]:
            raise ValueError(f"edge between {vertex1} and {vertex2} already exists.")

        insort(self.adjacency_list[vertex1], vertex2)
        if vertex1 != vertex2:
            insort(self.adjacency_list[vertex2], vertex1)

        self._nedges += 1

    @classmethod
    def from_kdtree(cls, kdtree):
        """Creates an undirected graph from a k-d tree."""
        graph = cls()
        node_index_map = {}
        node_count = 0

        # This function performs a depth-first traversal starting from the root of the KDTree.
        # It maintains a mapping of KDTree nodes to their corresponding indices in the graph.
        # As it traverses, it adds vertices for each unique node and edges for the connections between nodes.
        def traverse_and_add(node):
            nonlocal node_count
            if node is None:  # Stopping criteria
                return

            node_id = id(node)  # Need to use `id` since `Node` is unhashable
            if node_id not in node_index_map:
                node_index_map[node_id] = node_count
                node_count += 1
                graph.add_vertex()

            left_index = traverse_and_add(node.left)
            right_index = traverse_and_add(node.right)
            current_index = node_index_map[node_id]

            if left_index is not None:
                graph.add_edge(current_index, left_index)
            if right_index is not None:
                graph.add_edge(current_index, right_index)

            return current_index

        traverse_and_add(kdtree.root)
        return graph

    def to_networkx(self):
        """Converts an instance to a NetworkX graph.

        Returns:
            nx.Graph: A NetworkX graph representing the same undirected graph.
        """
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self.vertices)
        nx_graph.add_edges_from(self.edges)
        return nx_graph

    def __str__(self):
        strs = [f"UndirectedGraph with {self.nvertices} vertices and {self.nedges} edges:"]
        for vertex, adj_list in enumerate(self.adjacency_list):
            edge_str = ", ".join(f"{vertex} => {adj_vertex}" for adj_vertex in adj_list)
            strs.append(f" {vertex}: {edge_str}" if edge_str else f" {vertex}")

        return "\n".join(strs)
