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


    // 882. Reachable Nodes In Subdivided Graph
    // You are given an undirected graph (the "original graph") with n nodes labeled from 0 to n - 1.
    // You decide to subdivide each edge in the graph into a chain of nodes, with the number of new nodes varying between each edge.
    // The graph is given as a 2D array of edges where edges[i] = [ui, vi, cnti] indicates that there is an edge between nodes ui and vi in the original graph,
    // and cnti is the total number of new nodes that you will subdivide the edge into. Note that cnti == 0 means you will not subdivide the edge.
    // To subdivide the edge [ui, vi], replace it with (cnti + 1) new edges and cnti new nodes.
    // The new nodes are x1, x2, ..., xcnti, and the new edges are [ui, x1], [x1, x2], [x2, x3], ..., [xcnti-1, xcnti], [xcnti, vi].
    // In this new graph, you want to know how many nodes are reachable from the node 0, where a node is reachable if the distance is maxMoves or less.
    // Given the original graph and maxMoves, return the number of nodes that are reachable from node 0 in the new graph.
    // По-русски: дан граф с вершинами. Для каждого ребра графа дано колчиство вершин, которые нужно добавить на это ребро, разделив ими ребро.
    // Например, есть ребро [u, v, 2] - в итоговом графе имеем u-1-2-v, то есть 4 вершины и 3 ребра.
    // HARD, Dijkstra
    #region 882. Reachable Nodes In Subdivided Graph
    // Идея: НЕ строить новый граф с новыми вершинами, т.к. их много и это будет дорого по времени
    // Нужно считать cnti как расстояние между вершинами, или более точно, дополнительное количество шагов, которые нужно сделать, чтобы добраться из ui в vi
    // Сначала обходим исходный граф Дейкстрой, находим за какое количество шагов можно дойти до каждой вершины
    // Далее, имея количество шагов для каждой вершины, для каждого ребра считаем сколько новых вершин на этом ребре достижимы
    public int ReachableNodes(int[][] edges, int maxMoves, int n)
    {
        var adj = new List<(int, int)>[n];
        for (int i = 0; i < edges.Length; i++)
        {
            var a = edges[i][0];
            var b = edges[i][1];
            var c = edges[i][2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            if (adj[b] is null) adj[b] = new List<(int, int)>();
            adj[a].Add((b, c));
            adj[b].Add((a, c));
        }

        var dist = new int[n];
        for (int i = 1; i < n; i++)
        {
            dist[i] = int.MaxValue;
        }

        var heap = new PriorityQueue<int, int>();
        heap.Enqueue(0, 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var ca);
            //if (ca >= maxMoves) break; // с этим условием работает, но мне оно кажется каким-то мутным, поэтому закомментировал
            if (ca > dist[a] || adj[a] is null) continue;

            foreach (var (b, cb) in adj[a])
            {
                var newC = ca + cb + 1;
                if (dist[b] > newC)
                {
                    dist[b] = newC;
                    heap.Enqueue(b, newC);
                }
            }
        }

        var result = 0;
        for (int i = 0; i < edges.Length; i++) // считаем, до скольких новых вершин на каждом ребре мы можем дойти
        {
            var a = edges[i][0];
            var b = edges[i][1];
            var c = edges[i][2];
            var aLeft = 0;
            var bLeft = 0;
            if (dist[a] <= maxMoves) aLeft = maxMoves - dist[a];
            if (dist[b] <= maxMoves) bLeft = maxMoves - dist[b];

            result += Math.Min(bLeft + aLeft, c);
        }

        for (int i = 0; i < n; i++) // отдельно считаем, до каких исходных вершин мы дошли
        {
            if (dist[i] <= maxMoves) result++;
        }

        return result;
    }
    #endregion

    // 3341. Find Minimum Time to Reach Last Room I
    // There is a dungeon with n x m rooms arranged as a grid.
    // You are given a 2D array moveTime of size n x m, where moveTime[i][j] represents the minimum time in seconds after which the room opens and can be moved to.
    // You start from the room (0, 0) at time t = 0 and can move to an adjacent room. Moving between adjacent rooms takes exactly one second.
    // Return the minimum time to reach the room (n - 1, m - 1).
    // Two rooms are adjacent if they share a common wall, either horizontally or vertically.
    // Dijkstra's algorithm 
    #region 3341. Find Minimum Time to Reach Last Room I
    public int MinTimeToReach(int[][] moveTime)
    {
        var m = moveTime.Length;
        var n = moveTime[0].Length;
        var time = new int[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                time[i, j] = int.MaxValue;
            }
        }
        time[0, 0] = 0; // Не важно, во сколько откроется комната [0,0], можно из нее идти сразу, судя по тестам
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var heap = new PriorityQueue<(int i, int j), int>();
        //heap.Enqueue((0, 0), moveTime[0][0]); // Не важно, во сколько откроется комната [0,0], можно из нее идти сразу, судя по тестам
        heap.Enqueue((0, 0), 0); // Поэтому правильно так
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var t);
            if (t > time[a.i, a.j]) continue;
            if (a.i == m - 1 && a.j == n - 1) break;

            for (int d = 0; d < 4; d++)
            {
                var bi = a.i + di[d];
                var bj = a.j + dj[d];
                if (bi < 0 || bi >= m || bj < 0 || bj >= n) continue;
                var newTime = Math.Max(t, moveTime[bi][bj]) + 1; // +1 шаг требуется для перехода между комнатами
                if (time[bi, bj] > newTime)
                {
                    time[bi, bj] = newTime;
                    heap.Enqueue((bi, bj), newTime);
                }
            }
        }
        return time[m - 1, n - 1] == int.MaxValue
            ? -1
            : time[m - 1, n - 1];
    }
    #endregion

    // 2290. Minimum Obstacle Removal to Reach Corner
    // You are given a 0-indexed 2D integer array grid of size m x n. Each cell has one of two values:
    // - 0 represents an empty cell,
    // - 1 represents an obstacle that may be removed.
    // You can move up, down, left, or right from and to an empty cell.
    // Return the minimum number of obstacles to remove so you can move from the upper left corner (0, 0) to the lower right corner (m - 1, n - 1).
    // HARD
    // TODO: есть какое-то более оптимальное решение через BFS 1-0.
    #region 2290. Minimum Obstacle Removal to Reach Corner
    // Dijkstra's algorithm 
    public int MinimumObstacles(int[][] grid)
    {
        var m = grid.Length;
        var n = grid[0].Length;
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var dist = new int[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dist[i, j] = int.MaxValue;
            }
        }
        dist[0, 0] = 0;
        
        var heap = new PriorityQueue<(int i, int j), int>();
        heap.Enqueue((0, 0), dist[0, 0]);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var da);
            if (da > dist[a.i, a.j]) continue;
            if (a.i == m - 1 && a.j == n - 1) break;
            for (int d = 0; d < 4; d++)
            {
                var ii = a.i + di[d];
                var jj = a.j + dj[d];
                if (ii < 0 || ii >= m || jj < 0 || jj >= n) continue;
                var newDist = da + grid[ii][jj];
                if (dist[ii, jj] > newDist)
                {
                    dist[ii, jj] = newDist;
                    heap.Enqueue((ii, jj), newDist);
                }
            }
        }

        return dist[m - 1, n - 1];
    }
    #endregion

    // 2662. Minimum Cost of a Path With Special Roads
    // You are given an array start where start = [startX, startY] represents your initial position (startX, startY) in a 2D space.
    // You are also given the array target where target = [targetX, targetY] represents your target position (targetX, targetY).
    // The cost of going from a position (x1, y1) to any other position in the space (x2, y2) is |x2 - x1| + |y2 - y1|.
    // There are also some special roads. You are given a 2D array specialRoads where specialRoads[i] = [x1i, y1i, x2i, y2i, costi]
    // indicates that the ith special road goes in one direction from (x1i, y1i) to (x2i, y2i) with a cost equal to costi.
    // You can use each special road any number of times.
    // Return the minimum cost required to go from (startX, startY) to (targetX, targetY).
    // Dijkstra's algorithm 
    // Нестандартный Дейкстра: по-сути граф нужно строить налету
    // TODO: мой алгоритм слишом буквальный, есть какой-то попроще, говорят. Изучить.
    #region 2662. Minimum Cost of a Path With Special Roads
    // Идея: граф строится налету
    // Из текущей точки есть три варианта:
    // - Дойти сразу до target
    // - Дойти до ближайшей specialRoad
    // - Если текущая точка оказалась началом specialRoad, то можем также дойти до конца specialRoad
    public int MinimumCost(int[] start, int[] target, int[][] specialRoads)
    {
        var dist = new Dictionary<(int x, int y), int>();
        dist[(start[0], start[1])] = 0;
        var c0 = dst(start[0], start[1], target[0], target[1]);
        dist[(target[0], target[1])] = c0;

        var heap = new PriorityQueue<(int x, int y), int>(); 
        heap.Enqueue((start[0], start[1]), 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var curr, out var dcurr);
            if (dist.ContainsKey(curr) && dist[curr] < dcurr) continue;
            var tdst = dcurr + dst(curr.x, curr.y, target[0], target[1]); // Вариант пойти сразу в target
            if (tdst < dist[(target[0], target[1])])
            {
                dist[(target[0], target[1])] = tdst;
                heap.Enqueue((target[0], target[1]), tdst);
            }
            for (int i = 0; i < specialRoads.Length; i++)
            {
                var road = specialRoads[i];
                var ax = road[0];
                var ay = road[1];
                var bx = road[2];
                var by = road[3];
                var cost = road[4];
                if (ax == curr.x && ay == curr.y) // Вариант, когда текущая точка оказалась началом specialRoad
                {
                    var c = dcurr + cost;
                    if (!dist.ContainsKey((bx, by)) || dist[(bx, by)] > c)
                    {
                        dist[(bx, by)] = c;
                        heap.Enqueue((bx, by), c);
                    }
                }
                else // Вариант, когда можем дойти до ближайшей specialRoad
                {
                    var c = dcurr + dst(curr.x, curr.y, ax, ay);
                    if (!dist.ContainsKey((ax, ay)) || dist[(ax, ay)] > c)
                    {
                        dist[(ax, ay)] = c;
                        heap.Enqueue((ax, ay), c);
                    }
                }
            }
        }

        return dist[(target[0], target[1])];

        int dst(int x1, int y1, int x2, int y2)
        {
            return Math.Abs(x2 - x1) + Math.Abs(y2 - y1);
        }
    }
    #endregion

    // 3970. Shortest Path With At Most K Consecutive Identical Characters
    // You are given an integer n representing the number of nodes in a directed weighted graph, numbered from 0 to n - 1.
    // This is represented by a 2D integer array edges, where edges[i] = [ui, vi, wi] represents a directed edge from node ui to node vi with weight wi.
    // You are also given a string labels of length n, where labels[i] is the character assigned to node i, and an integer k.
    // Return the minimum total edge weight of a path from node 0 to node n - 1
    // such that the concatenation of the labels of the nodes along the path contains at most k consecutive identical characters.
    // If no valid path exists, return -1.
    #region 3970. Shortest Path With At Most K Consecutive Identical Characters
    // Идея: хранить только расстояние для каждой вершны недостаточно
    // Нужно хранить с привязкой к количеству одинаковых символов в текущем пути
    public int ShortestPath(int n, int[][] edges, string labels, int k)
    {
        var adj = new List<(int, int)>[n];
        foreach (var e in edges)
        {
            var a = e[0];
            var b = e[1];
            var w = e[2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            adj[a].Add((b, w));
        }

        var dist = new int[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                dist[i, j] = int.MaxValue;
            }
        }

        dist[0, 0] = 0;

        var heap = new PriorityQueue<(int a, int d, int c), int>();
        heap.Enqueue((0, 0, 0), 0);
        while (heap.Count > 0)
        {
            (var a, var da, var ka) = heap.Dequeue();
            if (dist[a, ka] < da || adj[a] is null) continue;
            // if (a == n - 1) return da; // Мутная оптимизация

            foreach (var (b, db) in adj[a])
            {
                var kb = labels[a] == labels[b] ? ka + 1 : 0;
                if (kb >= k) continue;
                var newD = da + db;
                if (newD < dist[b, kb])
                {
                    dist[b, kb] = newD;
                    heap.Enqueue((b, newD, kb), newD);
                }
            }
        }

        var result = int.MaxValue;
        for (int j = 0; j < k; j++)
        {
            result = Math.Min(result, dist[n - 1, j]);
        }
        return result == int.MaxValue ? -1 : result;
    }

    // Более оптимальная, но менее очевидная версия
    public int ShortestPath_earlyExit(int n, int[][] edges, string labels, int k)
    {
        var adj = new List<(int, int)>[n];
        foreach (var e in edges)
        {
            var a = e[0];
            var b = e[1];
            var w = e[2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            adj[a].Add((b, w));
        }

        var dist = new int[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                dist[i, j] = int.MaxValue;
            }
        }
        dist[0, 0] = 0;

        var heap = new PriorityQueue<(int a, int d, int c), int>();
        heap.Enqueue((0, 0, 0), 0);
        while (heap.Count > 0)
        {
            (var a, var da, var ka) = heap.Dequeue();
            if (dist[a, ka] < da) continue;
            if (a == n - 1) return da;
            if (adj[a] is null) continue;

            foreach (var (b, db) in adj[a])
            {
                var kb = labels[a] == labels[b] ? ka + 1 : 0;
                if (kb >= k) continue;
                var newD = da + db;
                if (newD < dist[b, kb])
                {
                    dist[b, kb] = newD;
                    heap.Enqueue((b, newD, kb), newD);
                }
            }
        }

        return -1;
    }
    #endregion

    // 505. The Maze II
    // There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1).
    // The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall.
    // When the ball stops, it could choose the next direction.
    // Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] and destination = [destinationrow, destinationcol],
    // return the shortest distance for the ball to stop at the destination.
    // If the ball cannot stop at destination, return -1.
    // The distance is the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included).
    // You may assume that the borders of the maze are all walls (see examples).
    // 
    // Dijkstra's algorithm, grid graph, неявный граф.
    // ВАЖНО: если граф неявный, не пытаться построить его полностью заранее, как правило лучше строить налету
    #region 505. The Maze II
    public int ShortestDistance(int[][] maze, int[] start, int[] destination)
    {
        var m = maze.Length;
        var n = maze[0].Length;
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var dist = new int[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dist[i, j] = int.MaxValue;
            }
        }
        dist[start[0], start[1]] = 0;
        var heap = new PriorityQueue<(int i, int j), int>();
        heap.Enqueue((start[0], start[1]), 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var da);
            if (da > dist[a.i, a.j]) continue;
            if (a.i == destination[0] && a.j == destination[1]) return da;

            for (int d = 0; d < 4; d++)
            {
                var bi = a.i;
                var bj = a.j;
                var db = da;
                var wi = a.i + di[d];
                var wj = a.j + dj[d];

                // find wall, катимся до стены в этом направлении
                while (wi >= 0 && wi < m && wj >= 0 && wj < n && maze[wi][wj] == 0)
                {
                    bi = wi;
                    bj = wj;
                    db++;
                    wi += di[d];
                    wj += dj[d];
                }

                if (db == da) continue; // не нашли следущую клетку в эту сторону (сразу уперлись в стену)

                if (db < dist[bi, bj])
                {
                    dist[bi, bj] = db;
                    heap.Enqueue((bi, bj), db);
                }
            }
        }

        return -1;
    }
    #endregion


    // 3650. Minimum Cost Path with Edge Reversals
    // You are given a directed, weighted graph with n nodes labeled from 0 to n - 1,
    // and an array edges where edges[i] = [ui, vi, wi] represents a directed edge from node ui to node vi with cost wi.
    // Each node ui has a switch that can be used at most once: when you arrive at ui and have not yet used its switch,
    // you may activate it on one of its incoming edges vi → ui reverse that edge to ui → vi and immediately traverse it.
    // The reversal is only valid for that single move, and using a reversed edge costs 2 * wi.
    // Return the minimum total cost to travel from node 0 to node n - 1. If it is not possible, return -1.
    // Dijkstra's algorithm
    // В условии непонятн написано: switch можно использовать для КАЖДОЙ вершины один раз, а не один раз за весь путь.
    #region 3650. Minimum Cost Path with Edge Reversals
    // Идея: switch представляет собой обратный список связанности с двойной стоимостью
    public int MinCost(int n, int[][] edges)
    {
        var adj = new List<(int i, int d)>[n];
        var rev = new List<(int i, int d)>[n];

        foreach (var e in edges)
        {
            var a = e[0];
            var b = e[1];
            var w = e[2];
            if (adj[a] is null) adj[a] = new List<(int i, int d)>();
            if (rev[b] is null) rev[b] = new List<(int i, int d)>();

            adj[a].Add((b, w));
            rev[b].Add((a, w + w)); // этот список у нас будет заменять switch
        }

        var dist = new int[n];
        for (int i = 1; i < n; i++)
        {
            dist[i] = int.MaxValue;
        }

        var heap = new PriorityQueue<int, int>();
        heap.Enqueue(0, 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var ad);
            if (ad > dist[a]) continue;
            if (a == n - 1) return ad;
            if (adj[a] is not null)
            {
                foreach (var b in adj[a])
                {
                    var newD = ad + b.d;
                    if (newD < dist[b.i])
                    {
                        dist[b.i] = newD;
                        heap.Enqueue(b.i, newD);
                    }
                }
            }
            if (rev[a] is not null) 
            {
                // Рассматриваем обратные ребра, т.е. используем switch
                // Мы так можем делать, т.к. в оптимальном пути не может быть два перехода из одной и той же вершины
                // Иначе мы имели бы цикл, который можно просто выкинуть
                // Таким образом мы гарантируем, что switch для этой вершины выполнится максимум один раз.
                foreach (var b in rev[a])
                {
                    var newD = ad + b.d;
                    if (newD < dist[b.i])
                    {
                        dist[b.i] = newD;
                        heap.Enqueue(b.i, newD);
                    }
                }
            }
        }

        return -1;
    }
    #endregion

    // 2093. Minimum Cost to Reach City With Discounts
    // A series of highways connect n cities numbered from 0 to n - 1.
    // You are given a 2D integer array highways where highways[i] = [city1i, city2i, tolli] indicates that there is a highway that connects city1i and city2i,
    // allowing a car to go from city1i to city2i and vice versa for a cost of tolli.
    // You are also given an integer discounts which represents the number of discounts you have.
    // You can use a discount to travel across the ith highway for a cost of tolli / 2 (integer division).
    // Each discount may only be used once, and you can only use at most one discount per highway.
    // Return the minimum total cost to go from city 0 to city n - 1, or -1 if it is not possible to go from city 0 to city n - 1.
    // Dijkstra's algorithm
    #region 2093. Minimum Cost to Reach City With Discounts
    public int MinimumCost(int n, int[][] highways, int discounts)
    {
        var adj = new List<(int, int)>[n];
        foreach (var h in highways)
        {
            var a = h[0];
            var b = h[1];
            var t = h[2];
            if (adj[a] is null) adj[a] = new List<(int, int)>();
            if (adj[b] is null) adj[b] = new List<(int, int)>();
            adj[a].Add((b, t));
            adj[b].Add((a, t));
        }

        var dist = new int[n, discounts + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= discounts; j++)
                dist[i, j] = int.MaxValue;
        }
        dist[0, 0] = 0;
        var heap = new PriorityQueue<(int i, int disc), int>();
        heap.Enqueue((0, 0), 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var ad);
            if (ad > dist[a.i, a.disc]) continue;
            if (a.i == n - 1) return ad;
            if (adj[a.i] is null) continue;

            foreach (var (bi, bd) in adj[a.i])
            {
                var newD = ad + bd;
                if (newD < dist[bi, a.disc])
                {
                    dist[bi, a.disc] = newD;
                    heap.Enqueue((bi, a.disc), newD);
                }
                if (a.disc < discounts)
                {
                    var newDDisc = ad + (bd / 2);
                    if (newDDisc < dist[bi, a.disc + 1])
                    {
                        dist[bi, a.disc + 1] = newDDisc;
                        heap.Enqueue((bi, a.disc + 1), newDDisc);
                    }
                }
            }
        }

        return -1;
    }
    #endregion

    // 3342. Find Minimum Time to Reach Last Room II
    // There is a dungeon with n x m rooms arranged as a grid.
    // You are given a 2D array moveTime of size n x m, where moveTime[i][j] represents the minimum time in seconds when you can start moving to that room.
    // You start from the room (0, 0) at time t = 0 and can move to an adjacent room.
    // Moving between adjacent rooms takes one second for one move and two seconds for the next, alternating between the two.
    // Return the minimum time to reach the room (n - 1, m - 1).
    // Two rooms are adjacent if they share a common wall, either horizontally or vertically.
    // Dijkstra's algorithm, grid
    #region 3342. Find Minimum Time to Reach Last Room II
    // TODO: в куче odd не нужен, т.к. в клетку можно прийти только за четное колчичество шагов
    // т.е. из клетки всегда стоимость перехода постоянная (либо всегда 1, либо всегда 2)
    // Это можно определить по (i+j)%2 + 1, то есть хранить и передавать odd необязательно
    public int MinTimeToReach2(int[][] moveTime)
    {
        var m = moveTime.Length;
        var n = moveTime[0].Length;

        var dist = new int[m, n, 2];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dist[i, j, 0] = int.MaxValue;
                dist[i, j, 1] = int.MaxValue;
            }
        }
        dist[0, 0, 0] = 0;

        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var heap = new PriorityQueue<(int i, int j, int odd), int>();
        heap.Enqueue((0, 0, 0), 0);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var a, out var at);
            if (at > dist[a.i, a.j, a.odd]) continue;
            if (a.i == m - 1 && a.j == n - 1) return at;

            for (int d = 0; d < 4; d++)
            {
                var bi = a.i + di[d];
                var bj = a.j + dj[d];
                if (bi < 0 || bi >= m || bj < 0 || bj >= n) continue;

                var bOdd = (a.odd + 1) % 2;
                var newT = Math.Max(at, moveTime[bi][bj]) + a.odd + 1;
                if (newT < dist[bi, bj, bOdd])
                {
                    dist[bi, bj, bOdd] = newT;
                    heap.Enqueue((bi, bj, bOdd), newT);
                }
            }
        }

        return -1;
    }
    #endregion

    // 2976. Minimum Cost to Convert String I
    // You are given two 0-indexed strings source and target, both of length n and consisting of lowercase English letters.
    // You are also given two 0-indexed character arrays original and changed, and an integer array cost,
    // where cost[i] represents the cost of changing the character original[i] to the character changed[i].
    // You start with the string source. In one operation, you can pick a character x from the string
    // and change it to the character y at a cost of z if there exists any index j such that cost[j] == z, original[j] == x, and changed[j] == y.
    // Return the minimum cost to convert the string source to the string target using any number of operations. If it is impossible to convert source to target, return -1.
    // Note that there may exist indices i, j such that original[j] == original[i] and changed[j] == changed[i].
    // Dijkstra's algorithm
    #region 2976. Minimum Cost to Convert String I
    // TODO: потенциальная оптимизация: после вычисления путей от 'a', например, до 'z', у нас уже есть все оптимальные пути от 'a' до других символов
    // Потенциально их сразу можно поместить в кеш.
    public long MinimumCost(string source, string target, char[] original, char[] changed, int[] cost)
    {
        var adj = new List<(int i, int c)>[26];
        for (int i = 0; i < original.Length; i++)
        {
            var a = original[i] - 'a';
            var b = changed[i] - 'a';
            var c = cost[i];

            if (adj[a] is null) adj[a] = new List<(int i, int c)>();
            adj[a].Add((b, c));
        }
        var cache = new Dictionary<(int, int), int>();
        long result = 0;
        for (int i = 0; i < source.Length; i++)
        {
            var start = source[i] - 'a';
            var end = target[i] - 'a';

            if (cache.ContainsKey((start, end)))
            {
                result += cache[(start, end)];
                continue;
            }

            var dist = new int[26];
            for (int j = 0; j < 26; j++)
            {
                dist[j] = int.MaxValue;
            }
            dist[start] = 0;
            var heap = new PriorityQueue<int, int>();
            heap.Enqueue(start, 0);
            var found = false;
            while (heap.Count > 0)
            {
                heap.TryDequeue(out var a, out var ad);
                if (ad > dist[a]) continue;
                if (a == end)
                {
                    cache[(start, end)] = ad;
                    result += ad;
                    found = true;
                    break;
                }
                if (adj[a] is null) continue;

                foreach (var b in adj[a])
                {
                    var newD = ad + b.c;
                    if (newD < dist[b.i])
                    {
                        dist[b.i] = newD;
                        heap.Enqueue(b.i, newD);
                    }
                }
            }

            if (!found)
            {
                return -1;
            }
        }

        return result;
    }
    #endregion


    // 684. Redundant Connection
    // In this problem, a tree is an undirected graph that is connected and has no cycles.
    // You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added.
    // The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed.
    // The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.
    // Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.
    #region 684. Redundant Connection
    public int[] FindRedundantConnection(int[][] edges)
    {
        var n = edges.Length;
        var children = new List<int>[n];
        var parent = new int[n];
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            children[i] = new List<int> { i };
        }

        foreach (var e in edges)
        {
            var a = e[0] - 1;
            var b = e[1] - 1;
            if (parent[a] == parent[b]) return e;
            var pa = parent[a];
            var pb = parent[b];

            if (children[pa].Count < children[pb].Count) // always move smaller list
                (pa, pb) = (pb, pa);

            if (children[pb].Count > 0)
            {
                foreach (var child in children[pb])
                {
                    parent[child] = pa;
                    children[pa].Add(child);
                }
                children[pb].Clear();
            }
        }

        return null;
    }
    #endregion

    // 1559. Detect Cycles in 2D Grid
    // Given a 2D array of characters grid of size m x n, you need to find if there exists any cycle consisting of the same value in grid.
    // A cycle is a path of length 4 or more in the grid that starts and ends at the same cell.
    // From a given cell, you can move to one of the cells adjacent to it - in one of the four directions (up, down, left, or right),
    // if it has the same value of the current cell.
    // Also, you cannot move to the cell that you visited in your last move.
    // For example, the cycle (1, 1) -> (1, 2) -> (1, 1) is invalid because from (1, 2) we visited (1, 1) which was the last visited cell.
    // Return true if any cycle of the same value exists in grid, otherwise, return false.
    #region 1559. Detect Cycles in 2D Grid
    public bool ContainsCycle(char[][] grid)
    {
        var m = grid.Length;
        var n = grid[0].Length;
        var visited = new bool[m, n];
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var queue = new Queue<(int i, int j)>();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (visited[i, j]) continue;

                queue.Enqueue((i, j));
                while (queue.Count > 0)
                {
                    var (ai, aj) = queue.Dequeue();
                    if (visited[ai, aj]) return true;
                    visited[ai, aj] = true;

                    for (int d = 0; d < 4; d++)
                    {
                        var bi = ai + di[d];
                        var bj = aj + dj[d];
                        if (bi < 0 || bi >= m || bj < 0 || bj >= n) continue;
                        if (visited[bi, bj]) continue;
                        if (grid[bi][bj] == grid[ai][aj])
                        {
                            queue.Enqueue((bi, bj));
                        }
                    }
                }
            }
        }

        return false;
    }
    #endregion
}
