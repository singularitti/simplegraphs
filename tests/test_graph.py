import unittest

import numpy as np
from simplegraphs.graph import UndirectedGraph


class TestUndirectedGraph(unittest.TestCase):
    def test_initialization(self):
        self.assertEqual(UndirectedGraph([[1], [0]]).adjacency_list, [[1], [0]])

    def test_from_edges(self):
        # Create a graph from a set of edges and a specified number of vertices
        edges = {(0, 1), (1, 2)}
        nvertices = 4  # Including a vertex (3) with no edges
        graph = UndirectedGraph.from_edges(edges, nvertices)
        # Test if the graph has the correct number of vertices
        self.assertEqual(graph.nvertices, nvertices)
        # Test if the graph has the correct edges
        self.assertEqual(graph.edges, edges)
        # Test if the graph's adjacency list is correctly formed
        expected_adj_list = [[1], [0, 2], [1], []]  # Vertex 3 is isolated
        self.assertEqual(graph.adjacency_list, expected_adj_list)
        # Test if the number of edges is correct
        self.assertEqual(graph.nedges, len(edges))
        with self.assertRaises(IndexError):
            UndirectedGraph.from_edges({(0, 1)}, 1)
        with self.assertRaises(TypeError):
            UndirectedGraph.from_edges({(0, 0.5)}, 2)

    def test_from_adjacency_matrix(self):
        # From https://stackoverflow.com/a/62241950/3260253
        matrix = np.asarray(
            [
                [0, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1, 0],
            ]
        )
        np.testing.assert_allclose(matrix, matrix.T)  # Test if matrix is symmetric
        graph = UndirectedGraph.from_adjacency_matrix(matrix)
        self.assertEqual(
            graph.adjacency_list,
            [[1, 3], [0, 2, 4], [1, 5], [0, 4, 5], [1, 3, 5], [2, 3, 4]],
        )
        self.assertEqual(graph.nvertices, 6)
        self.assertEqual(graph.nedges, 8)
        self.assertEqual(
            graph.edges, {(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 5), (4, 5)}
        )

    def test_empty_graph(self):
        graph = UndirectedGraph()
        self.assertEqual(graph.nvertices, 0)
        self.assertEqual(graph.nedges, 0)
        self.assertEqual(graph.vertices, [])
        self.assertEqual(graph.edges, set())
        self.assertEqual(graph.connected_components, set())
        with self.assertRaises(IndexError):
            graph.dfs(0)
            graph.bfs(0)

    def test_add_vertex(self):
        graph = UndirectedGraph()
        graph.add_vertex()
        self.assertEqual(graph.adjacency_list, [[]])

    def test_add_edge(self):
        graph = UndirectedGraph()
        graph.add_vertex()
        graph.add_vertex()
        graph.add_edge(0, 1)
        self.assertEqual(graph.adjacency_list, [[1], [0]])

    def test_weighted_graph(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        self.assertEqual(graph.vertices, [0, 1, 2])
        self.assertEqual(graph.nvertices, 3)
        self.assertEqual(graph.edges, {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(graph.nedges, 3)

    def test_self_loop(self):
        graph = UndirectedGraph()
        graph.add_vertex()
        graph.add_edge(0, 0)  # Adding a self-loop
        self.assertEqual(graph.nedges, 1)
        self.assertEqual(graph.edges, {(0, 0)})
        with self.assertRaises(ValueError):
            graph.add_edge(0, 0)  # Edge already exists

    def test_dfs(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        visited = graph.dfs(0)
        self.assertEqual(visited, {0, 1, 2})

    def test_bfs(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        visited = graph.bfs(0)
        self.assertEqual(visited, {0, 1, 2})

    def test_connected_components(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        components = graph.connected_components
        self.assertEqual(components, {frozenset({0, 1, 2})})

    def test_modification(self):
        graph = UndirectedGraph.from_edges({(0, 1)}, 2)
        graph.add_vertex()
        graph.add_edge(1, 2)
        self.assertEqual(graph.nvertices, 3)
        self.assertEqual(graph.nedges, 2)
        self.assertEqual(graph.connected_components, {frozenset({0, 1, 2})})

    def test_str(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        self.assertIn("UndirectedGraph with 3 vertices and 3 edges:", str(graph))

    def test_error_handling(self):
        graph = UndirectedGraph([[1, 2], [0, 2], [0, 1]])
        with self.assertRaises(IndexError):
            graph.add_edge(0, 3)

        with self.assertRaises(ValueError):
            graph.add_edge(0, 1)  # Edge already exists
            graph.add_edge(1, 0)

    def test_large_graph(self):
        n = 1000
        edges = {(i, i + 1) for i in range(n - 1)}
        graph = UndirectedGraph.from_edges(edges, n)
        self.assertEqual(graph.nvertices, n)
        self.assertEqual(graph.nedges, n - 1)


if __name__ == "__main__":
    unittest.main()
