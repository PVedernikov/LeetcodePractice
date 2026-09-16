namespace LeetcodePreapare;

public class LeetCodeGeneral
{
    // 24. Swap Nodes in Pairs
    // Given a linked list, swap every two adjacent nodes and return its head.
    // You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
    // Example 1:
    // Input:  [1,2,3,4]
    // Output: [2,1,4,3]
    // Linked List
    #region 24. Swap Nodes in Pairs
    public ListNode SwapPairs(ListNode head)
    {
        if (head is null || head.next is null) return head;
        ListNode prev = null;
        var first = head;
        var result = head.next;
        while (first is not null)
        {
            var second = first.next;
            if (second is not null)
            {
                first.next = second.next;
                second.next = first;
                if (prev is not null)
                {
                    prev.next = second;
                }
                prev = first;
            }
            first = first.next;
        }
        return result;
    }
    #endregion


    // 28. Find the Index of the First Occurrence in a String
    // Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
    // TODO: implement KMP algorithm
    #region 28. Find the Index of the First Occurrence in a String
    // Broot force solution
    public int StrStr(string haystack, string needle)
    {
        var n1 = haystack.Length;
        var n2 = needle.Length;

        if (n1 < n2) return -1;

        for (int i = 0; i < n1; i++)
        {
            var valid = true;
            for (int j = 0; j < n2; j++)
            {
                if (i + j >= n1 || haystack[i + j] != needle[j])
                {
                    valid = false;
                    break;
                }
            }

            if (valid) return i;
        }

        return -1;
    }
    #endregion

    // 343. Integer Break
    // Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.
    // Return the maximum product you can get.
    // Example 1: Input: n = 2 Output: 1 Explanation: 2 = 1 + 1, 1 × 1 = 1.
    // Example 2: Input: n = 10 Output: 36 Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
    #region 343. Integer Break
    // 1D DP solution
    // Time complexity: O(n^2)
    public int IntegerBreak_DP(int n)
    {
        var dp = new int[n + 1];
        return dfs(n);

        int dfs(int a)
        {
            if (a == 1) return 1;
            if (dp[a] > 0) return dp[a];
            // Because k >= 2, we can't take n, because in this case the we have only one number
            // But if a < n, we can take a, because in this case we have at least two numbers, a and n - a
            var result = a < n ? a : 0; 
            for (int i = 1; i < a; i++)
            {
                result = Math.Max(result, i * dfs(a - i));
            }
            dp[a] = result;
            return result;
        }
    }

    // Greedy solution
    // Time complexity: O(n)
    // Идея: лучше всего разбивать число на как можно больше троек, но нельзя оставлять остаток 1
    // 7 = 3 + 4, 3 * 4 = 12
    // 8 = 3 + 3 + 2, 3 * 3 * 2 = 18
    // 9 = 3 + 3 + 3, 3 * 3 * 3 = 27
    // 10 = 3 + 3 + 4, 3 * 3 * 4 = 36
    public int IntegerBreak(int n)
    {
        if (n == 2) return 1;
        if (n == 3) return 2;
        if (n == 4) return 4;

        var result = 1;
        while (n > 4)
        {
            result *= 3;
            n -= 3;
        }

        return result * n;
    }
    #endregion

    // 1514. Path with Maximum Probability
    // You are given an undirected weighted graph of n nodes (0-indexed), represented by an edge list where edges[i] = [a, b] is an undirected edge
    // connecting the nodes a and b with a probability of success of traversing that edge succProb[i].
    // Given two nodes start and end, find the path with the maximum probability of success to go from start to end and return its success probability.
    // If there is no path from start to end, return 0. Your answer will be accepted if it differs from the correct answer by at most 1e-5.
    // Note: вероятность успеха пути - это произведение вероятностей успеха всех рёбер на этом пути.
    // Almost classic Dijkstra's algorithm
    #region 1514. Path with Maximum Probability
    public double MaxProbability(int n, int[][] edges, double[] succProb, int start_node, int end_node)
    {
        var adj = new List<(int i, double p)>[n];
        for (int i = 0; i < edges.Length; i++)
        {
            var a = edges[i][0];
            var b = edges[i][1];
            if (adj[a] is null) adj[a] = new List<(int i, double p)>();
            if (adj[b] is null) adj[b] = new List<(int i, double p)>();
            adj[a].Add((b, succProb[i]));
            adj[b].Add((a, succProb[i]));
        }
        var probability = new double[n];
        var heap = new PriorityQueue<int, double>();
        heap.Enqueue(start_node, -1.0);
        probability[start_node] = 1.0;
        while (heap.Count > 0)
        {
            //var a = heap.Dequeue();
            heap.TryDequeue(out int a, out double p);
            p = -p;
            // оптимизация: если мы уже нашли путь с большей вероятностью, нет смысла идти дальше
            // но без этого тоже работает
            if (probability[a] > p) continue; 

            if (a == end_node) return probability[a]; // early exit

            if (adj[a] is null) continue;

            foreach (var b in adj[a])
            {
                var newP = probability[a] * b.p;
                if (probability[b.i] < newP)
                {
                    probability[b.i] = newP;
                    heap.Enqueue(b.i, -probability[b.i]);
                }
            }
        }

        return probability[end_node];
    }
    #endregion

    // 2642. Design Graph With Shortest Path Calculator
    // There is a directed weighted graph that consists of n nodes numbered from 0 to n - 1.
    // The edges of the graph are initially represented by the given array edges where edges[i] = [fromi, toi, edgeCosti]
    // meaning that there is an edge from fromi to toi with the cost edgeCosti.
    // Implement the Graph class:
    // - Graph(int n, int[][] edges) initializes the object with n nodes and the given edges.
    // - addEdge(int[] edge) adds an edge to the list of edges where edge = [from, to, edgeCost].
    //   It is guaranteed that there is no edge between the two nodes before adding this one.
    // - int shortestPath(int node1, int node2) returns the minimum cost of a path from node1 to node2.
    //   If no path exists, return -1. The cost of a path is the sum of the costs of the edges in the path.
    // HARD
    // Classic Dijkstra's algorithm
    #region 2642. Design Graph With Shortest Path Calculator
    public class Graph
    {
        private int _n;
        private List<(int, int)>[] _adj;
        public Graph(int n, int[][] edges)
        {
            _n = n;
            _adj = new List<(int, int)>[n];
            for (int i = 0; i < edges.Length; i++)
            {
                var a = edges[i][0];
                var b = edges[i][1];
                var cost = edges[i][2];
                if (_adj[a] is null) _adj[a] = new List<(int, int)>();

                _adj[a].Add((b, cost));
            }
        }

        public void AddEdge(int[] edge)
        {
            var a = edge[0];
            var b = edge[1];
            var cost = edge[2];
            if (_adj[a] is null) _adj[a] = new List<(int, int)>();
            _adj[a].Add((b, cost));
        }

        public int ShortestPath(int node1, int node2)
        {
            var cost = new int[_n];
            for (int i = 0; i < _n; i++)
            {
                cost[i] = int.MaxValue;
            }
            cost[node1] = 0;
            var heap = new PriorityQueue<int, int>();
            heap.Enqueue(node1, 0);
            while (heap.Count > 0)
            {
                heap.TryDequeue(out int a, out int aCost);
                if (aCost > cost[a]) continue;
                if (a == node2) return aCost;
                if (_adj[a] is null) continue;

                foreach (var (b, costB) in _adj[a])
                {
                    var newCost = cost[a] + costB;
                    if (cost[b] > newCost)
                    {
                        cost[b] = newCost;
                        heap.Enqueue(b, newCost);
                    }
                }
            }

            return cost[node2] == int.MaxValue ? -1 : cost[node2];
        }
    }
    #endregion

    // 3112. Minimum Time to Visit Disappearing Nodes
    // There is an undirected graph of n nodes.
    // You are given a 2D array edges, where edges[i] = [ui, vi, lengthi] describes an edge between node ui and node vi with a traversal time of lengthi units.
    // Additionally, you are given an array disappear, where disappear[i] denotes the time when the node i disappears from the graph and you won't be able to visit it.
    // Note that the graph might be disconnected and might contain multiple edges.
    // Return the array answer, with answer[i] denoting the minimum units of time required to reach node i from node 0.
    // If node i is unreachable from node 0 then answer[i] is -1.
    #region 3112. Minimum Time to Visit Disappearing Nodes
    // Classic Dijkstra's algorithm с доп условием
    public int[] MinimumTime(int n, int[][] edges, int[] disappear)
    {
        var adj = new List<(int, int)>[n];
        for (int i = 0; i < edges.Length; i++)
        {
            var a = edges[i][0];
            var b = edges[i][1];
            var t = edges[i][2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            if (adj[b] is null) adj[b] = new List<(int, int)>();
            adj[a].Add((b, t));
            adj[b].Add((a, t));
        }

        var times = new int[n];
        for (int i = 1; i < n; i++)
        {
            times[i] = int.MaxValue;
        }
        var heap = new PriorityQueue<int, int>();

        heap.Enqueue(0, 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out int a, out int t);
            if (adj[a] is null) continue;
            if (times[a] < t) continue;

            foreach (var (b, tb) in adj[a])
            {
                var newT = times[a] + tb;
                if (newT < disappear[b] && newT < times[b])
                {
                    times[b] = newT;
                    heap.Enqueue(b, newT);
                }
            }
        }

        for (int i = 1; i < n; i++)
        {
            if (times[i] == int.MaxValue)
            {
                times[i] = -1;
            }
        }
        return times;
    }
    #endregion

    // 1976. Number of Ways to Arrive at Destination
    #region 1976. Number of Ways to Arrive at Destination
    // Dijkstra's algorithm with counting the number of ways to reach each node
    // TODO
    #endregion

    // 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
    // There are n cities numbered from 0 to n-1. Given the array edges where edges[i] = [fromi, toi, weighti]
    // represents a bidirectional and weighted edge between cities fromi and toi, and given the integer distanceThreshold.
    // Return the city with the smallest number of cities that are reachable through some path and whose distance is at most distanceThreshold,
    // If there are multiple such cities, return the city with the greatest number.
    // Notice that the distance of a path connecting cities i and j is equal to the sum of the edges' weights along that path.
    #region 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
    // Dijkstra's algorithm 
    public int FindTheCity(int n, int[][] edges, int distanceThreshold)
    {
        var adj = new List<(int, int)>[n];
        for (int i = 0; i < edges.Length; i++)
        {
            var a = edges[i][0];
            var b = edges[i][1];
            var t = edges[i][2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            if (adj[b] is null) adj[b] = new List<(int, int)>();
            adj[a].Add((b, t));
            adj[b].Add((a, t));
        }
        var min = int.MaxValue;
        var index = -1;
        for (int i = n - 1; i >= 0; i--)
        {
            var times = new int[n];
            for (int j = 0; j < n; j++)
            {
                times[j] = int.MaxValue;
            }
            times[i] = 0;

            var heap = new PriorityQueue<int, int>();
            heap.Enqueue(i, 0);
            while (heap.Count > 0)
            {
                heap.TryDequeue(out var a, out var ta);
                if (ta > times[a] || ta > distanceThreshold || adj[a] is null) continue;

                foreach (var (b, tb) in adj[a])
                {
                    var newT = ta + tb;
                    if (newT < times[b])
                    {
                        times[b] = newT;
                        heap.Enqueue(b, newT);
                    }
                }
            }

            var count = 0;
            for (int j = 0; j < n; j++)
            {
                if (j != i && times[j] <= distanceThreshold)
                {
                    count++;
                }
            }

            if (count < min)
            {
                min = count;
                index = i;
            }
        }

        return index;
    }
    #endregion

}
