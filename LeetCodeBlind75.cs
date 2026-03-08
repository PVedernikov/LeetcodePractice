using System.Collections.Generic;
using System.IO;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace LeetcodePreapare;

// https://leetcode.com/problem-list/r3q9lspc/
public static class LeetCodeBlind75
{
    // #1
    // 128. Longest Consecutive Sequence
    // Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
    #region 128. Longest Consecutive Sequence
    public static int LongestConsecutive(int[] nums)
    {
        if (nums.Length == 0) return 0;

        var uniqueNums = new HashSet<int>();
        for (int i = 0; i < nums.Length; i++)
        {
            uniqueNums.Add(nums[i]);
        }

        var result = 0;
        foreach (var num in uniqueNums)
        {
            if (!uniqueNums.Contains(num - 1))
            {
                var length = 1;
                while (uniqueNums.Contains(num + length))
                {
                    length++;
                }
                result = Math.Max(result, length);
            }
        }

        return result;
    }
    #endregion

    // #2
    // 1. Two Sum
    // HashSet
    #region 1. Two Sum
    public static int[] TwoSum(int[] nums, int target)
    {
        var expecting = new Dictionary<int, int>();

        for (int i = 0; i < nums.Length; i++)
        {
            if (expecting.ContainsKey(nums[i]))
            {
                return [expecting[nums[i]], i];
            }
            expecting[target - nums[i]] = i;
        }

        return Array.Empty<int>();
    }
    #endregion

    // #3
    // 3. Longest Substring Without Repeating Characters
    // Sliding Window, HashSet
    // Given a string s, find the length of the longest substring without duplicate characters.
    #region LengthOfLongestSubstring
    public static int LengthOfLongestSubstring(string s)
    {
        if (s.Length == 0) return 0;

        var left = 0;
        var right = 1;
        var result = 1;

        var symbols = new HashSet<char>();
        symbols.Add(s[left]);
        while (right < s.Length)
        {
            while (right < s.Length && !symbols.Contains(s[right]))
            {
                symbols.Add(s[right]);
                result = Math.Max(result, symbols.Count);
                right++;
            }

            symbols.Remove(s[left]);
            left++;
            if (left >= right)
            {
                right = left + 1;
            }
        }
        return result;
    }

    public static int LengthOfLongestSubstringClener(string s)
    {
        if (s.Length <= 0) return s.Length;

        var left = 0;
        var result = 0;

        var symbols = new HashSet<char>();
        for (int right = 0; right < s.Length; right++)
        {
            while (symbols.Contains(s[right]))
            {
                symbols.Remove(s[left]);
                left++;
            }

            symbols.Add(s[right]);
            result = Math.Max(result, right - left + 1);
        }
        return result;
    }
    #endregion

    // #4
    // 5. Longest Palindromic Substring
    // Given a string s, return the longest palindromic substring in s.
    // Expand Around Center
    public static string LongestPalindrome(string s)
    {
        if (s.Length <= 1) return s;

        var start = 0;
        var length = 1;
        for (int i = 0; i < s.Length; i++)
        {
            var l = i;
            var r = i;
            while (r < s.Length - 1 && s[i] == s[r + 1])
            {
                r++;
            }
            i = r; // Пропустить одинаковые символы, мы их рассмотрим в этой иетерации

            while (l > 0 && r < s.Length - 1 && s[l - 1] == s[r + 1])
            {
                l--;
                r++;
            }
            var len = r - l + 1;
            if (len > length)
            {
                start = l;
                length = len;
            }
        }

        return s.Substring(start, length);
    }

    // #5
    // 133. Clone Graph
    // Given a reference of a node in a connected undirected graph.
    // Return a deep copy (clone) of the graph.
    // Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
    // DFS, BFS, HashMap
    #region CloneGraph DFS
    public static Node133 CloneGraph(Node133 node)
    {
        if (node is null)
        {
            return null;
        }
        return GetClone(node, new Dictionary<Node133, Node133>());
    }

    private static Node133 GetClone(Node133 node, Dictionary<Node133, Node133> cloned)
    {
        if (cloned.TryGetValue(node, out var clone))
        {
            return clone;
        }

        var newNode = new Node133(node.val, new List<Node133>());
        cloned[node] = newNode;

        if (node.neighbors is not null)
        {
            foreach (var neighbor in node.neighbors)
            {
                newNode.neighbors.Add(GetClone(neighbor, cloned));
            }
        }
        return newNode;
    }
    #endregion
    #region CloneGraph BFS
    public static Node133 CloneGraphBFS(Node133 node)
    {
        if (node is null)
        {
            return null;
        }

        var cloned = new Dictionary<Node133, Node133>();
        var queue = new Queue<Node133>();
        cloned[node] = new Node133(node.val);
        queue.Enqueue(node);

        while (queue.Count > 0)
        {
            var origNode = queue.Dequeue();
            var newNode = cloned[origNode];

            foreach (var neighbor in origNode.neighbors)
            {
                if (!cloned.ContainsKey(neighbor))
                {
                    var clonedNeighbor = new Node133(neighbor.val);
                    cloned[neighbor] = clonedNeighbor;
                    newNode.neighbors.Add(clonedNeighbor);
                    queue.Enqueue(neighbor);
                }
                else
                {
                    newNode.neighbors.Add(cloned[neighbor]);
                }
            }
        }

        return cloned[node];
    }
    #endregion

    // #6
    // 261. Graph Valid Tree
    // You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges
    // where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.
    // Return true if the edges of the given graph make up a valid tree, and false otherwise.
    // 
    // According to the definition of tree on Wikipedia: “a tree is an undirected graph in which any two vertices are connected by exactly one path.
    // In other words, any connected graph without simple cycles is a tree.”
    // DFS
    // O(n + m) time complexity, O(n + m) space complexity, where n - number of vertices, m - number of edges
    #region 261. Graph Valid Tree
    // My first solution
    // Идея: обходим граф в глубину, проверяя на циклы через visited
    // Кроме того считаем количество посещенных вершин, в конце проверяем, что мы посетили все вершины, т.е. граф связный, и при этом не нашли циклов, значит это дерево
    public static bool ValidTree(int n, int[][] edges)
    {
        if (edges.Length != n - 1) return false; // If it's tree, it must have exactly n - 1 edges

        var adj = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new List<int>();
        }

        for (int i = 0; i < edges.Length; i++)
        {
            adj[edges[i][0]].Add(edges[i][1]);
            adj[edges[i][1]].Add(edges[i][0]);
        }

        var visited = new bool[n];
        var visitedCount = 0;
        var stack = new Stack<(int, int)>();
        stack.Push((0, -1));
        while (stack.Count > 0)
        {
            (var i, var parent) = stack.Pop();
            if (visited[i]) return false;
            visited[i] = true;
            visitedCount++;
            foreach (var j in adj[i])
            {
                if (j != parent)
                {
                    stack.Push((j, i));
                }
            }
        }

        return visitedCount == n; // visited all vertices, so graph is connected, and we didn't find cycles, so it's a tree
    }

    // Другой вариант
    // Идея и отличие от первого: мы не проверяем циклы при обходе графа, т.к. если количество ребер правильное n-1,
    // то циклы невозможны, при условии что граф связный, а мы проверяем связность в конце, через visitedCount == n (то есть мы посетили ровно n вершин)
    // Если бы в графе был цикл, это значит, что граф не связанный, то мы бы тогда посетили меньше вершин, потому что не попали бы в какую-то часть графа,
    // которая не связана с остальными вершинами
    public static bool ValidTree_1(int n, int[][] edges)
    {
        if (edges.Length != n - 1) return false; // If it's tree, it must have exactly n - 1 edges

        var adj = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new List<int>();
        }

        for (int i = 0; i < edges.Length; i++)
        {
            adj[edges[i][0]].Add(edges[i][1]);
            adj[edges[i][1]].Add(edges[i][0]);
        }

        var visited = new bool[n];
        var visitedCount = 0;
        var stack = new Stack<int>();
        stack.Push((0));
        while (stack.Count > 0)
        {
            var i = stack.Pop();
            if (visited[i]) continue;
            visited[i] = true;
            visitedCount++;
            foreach (var j in adj[i])
            {
                stack.Push(j);
            }
        }

        return visitedCount == n; // visited all vertices, so graph is connected
    }
    #endregion

    // #7
    // 647. Palindromic Substrings
    // Given a string s, return the number of palindromic substrings in it.
    // Expand Around Center
    public static int CountSubstrings(string s)
    {
        var n = s.Length;
        if (n <= 1)
        {
            return n;
        }

        var result = 0;
        for (int i = 0; i < n; i++)
        {
            var l = i;
            var r = i;
            while (r < n - 1 && s[r] == s[r + 1])
            {
                r++;
            }
            i = r;
            var centerCount = r - l + 1;

            // Substrings in a string with length = n formula:
            // n * (n + 1) / 2
            result += (centerCount * (centerCount + 1)) / 2;

            while (l > 0 && r < n - 1 && s[l - 1] == s[r + 1])
            {
                result++;
                l--;
                r++;
            }
        }

        return result;
    }

    // #8
    // 11. Container With Most Water
    // You are given an integer array height of length n.
    // There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
    // Find two lines that together with the x-axis form a container, such that the container contains the most water.
    // Two Pointers, Array, Greedy
    public static int MaxArea(int[] height)
    {
        var n = height.Length;
        if (n <= 1)
        {
            return 0;
        }

        var result = 0;
        var l = 0;
        var r = n - 1;
        while (l < r)
        {
            var minHeight = Math.Min(height[l], height[r]);
            var area = (r - l) * minHeight;
            result = Math.Max(result, area);

            if (height[l] < height[r])
            {
                l++;
            }
            else
            {
                r--;
            }
        }

        return result;
    }

    // #9
    // 139. Word Break
    // Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
    // Top-Down DP, Memoization
    #region 139. Word Break
    public static bool WordBreak(string s, IList<string> wordDict)
    {
        return IsValidWordBreak(s, 0, wordDict, new Dictionary<int, bool>());
    }

    private static bool IsValidWordBreak(string s, int start, IList<string> wordDict, Dictionary<int, bool> cache)
    {
        var n = s.Length;
        if (start == n) return true;
        if (cache.ContainsKey(start)) return cache[start];

        var len = n - start;
        foreach (var word in wordDict)
        {
            if (word.Length > len)
            {
                continue;
            }

            var valid = true;
            for (int i = 0; i < word.Length; i++)
            {
                if (word[i] != s[start + i])
                {
                    valid = false;
                    break;
                }
            }

            if (valid && IsValidWordBreak(s, start + word.Length, wordDict, cache))
            {
                cache[start] = true;
                return true;
            }
        }

        cache[start] = false;
        return false;
    }
    #endregion

    // #10
    // 141. Linked List Cycle
    // Given head, the head of a linked list, determine if the linked list has a cycle in it.
    // Floyd's Tortoise and Hare
    // Linked List, Cycle Detection, Fast & Slow Pointers
    public static bool HasCycle(ListNode head)
    {
        var slow = head;
        var fast = head;

        while (fast is not null && fast.next is not null)
        {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast) return true;
        }

        return false;
    }

    // #11
    // 268. Missing Number
    // Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
    public static int MissingNumber(int[] nums)
    {
        var n = nums.Length;
        var result = ((n + 1) * n) / 2;

        for (int i = 0; i < n; i++)
        {
            result -= nums[i];
        }

        return result;
    }

    // #12
    // 15. 3Sum
    // Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    // Notice that the solution set must not contain duplicate triplets.
    // Sort, Two Pointers
    public static IList<IList<int>> ThreeSum(int[] nums)
    {
        Array.Sort(nums);

        IList<IList<int>> result = new List<IList<int>>();

        for (int i = 0; i < nums.Length - 2; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;

            var l = i + 1;
            var r = nums.Length - 1;
            while (l < r)
            {
                var sum = nums[i] + nums[l] + nums[r];
                if (sum < 0)
                {
                    l++;
                }
                else if (sum > 0)
                {
                    r--;
                }
                else
                {
                    result.Add(new List<int> { nums[i], nums[l], nums[r] });
                    l++;
                    while (nums[l] == nums[l - 1] && l < r)
                    {
                        l++;
                    }
                }
            }
        }
        return result;
    }

    // #13
    // 143. Reorder List
    // You are given the head of a singly linked-list. L0 → L1 → … → Ln - 1 → Ln
    // Reorder the list to be on the following form: L0 → Ln → L1 → L(n - 1) → L2 → L(n - 2) → …
    #region 143. Reorder List
    // O(n) time complexity, O(1) memory complexity
    // Идея: найти центр списка (fast & slow указатели), развернуть вторую половину, затем слить две половины
    public static void ReorderList(ListNode head)
    {
        if (head is null || head.next is null) return;

        // 1. Find the middle of the list with fast & slow pointers
        var slow = head;
        var fast = head.next;
        while (fast is not null && fast.next is not null)
        {
            slow = slow.next;
            fast = fast.next.next;
        }

        // slow is the middle of the list now, slow.next is next half
        // 2. Reverse the second half of the list
        var dummyHead = new ListNode();
        var current = slow.next;
        slow.next = null;
        while (current is not null)
        {
            var tail = dummyHead.next;
            dummyHead.next = current;
            current = current.next;
            dummyHead.next.next = tail;
        }

        // 3. Merge two halves
        var top = head;
        var bottom = dummyHead.next;
        while (bottom is not null)
        {
            var tmp = top.next;
            top.next = bottom;
            bottom = bottom.next;
            top.next.next = tmp;
            top = tmp;
        }
    }

    // O(n) time complexity, O(n) memory complexity
    // Идея: положить все элементы в стек, затем проходясь по списку, вставлять элементы из стека между элементами списка пока не дойдем до середины списка
    // Середина определяется по совпадению текущего элемента и элемента на вершине стека (для нечетного количества элементов)
    // или по совпадению следующего элемента и элемента на вершине стека (для четного количества элементов)
    // НЕ ОПТИМАЛЬНО ПО ПАМЯТИ
    public static void ReorderListOn(ListNode head)
    {
        if (head is null) return;

        var current = head;
        var stack = new Stack<ListNode>();
        while (current is not null)
        {
            stack.Push(current);
            current = current.next;
        }

        current = head;
        while (current is not null && stack.Count > 0)
        {
            var next = stack.Pop();

            // Два if для четных и нечетных длин списков
            if (current == next)
            {
                current.next = null;
                break;
            }

            if (current.next == next)
            {
                current.next.next = null;
                break;
            }

            var tmp = current.next;
            current.next = next;
            next.next = tmp;
            current = next.next;
        }
    }
    #endregion

    // #14
    // 269. Alien Dictionary
    // There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.
    // You are given a list of strings words from the alien language's dictionary. Now it is claimed that the strings in words are sorted lexicographically by the rules of this new language.
    // If this claim is incorrect, and the given arrangement of string in words cannot correspond to any order of letters, return "".
    // Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there are multiple solutions, return any of them.
    // 
    // Topological Sort, DFS, Graph
    #region 269. Alien Dictionary

    public static string AlienOrder(string[] words)
    {
        var n = words.Length;
        if (n == 0) return string.Empty;

        var adj = new Dictionary<char, HashSet<char>>();
        for (int i = 0; i < n; i++)
        {
            var len = words[i].Length;
            for (int k = 0; k < len; k++)
            {
                if (!adj.ContainsKey(words[i][k]))
                {
                    adj[words[i][k]] = new HashSet<char>();
                }
            }
        }

        for (int i = 1; i < n; i++)
        {
            var word1 = words[i - 1];
            var word2 = words[i];
            var len1 = word1.Length;
            var len2 = word2.Length;
            var len = Math.Min(len1, len2);
            for (int k = 0; k < len; k++)
            {
                if (word1[k] == word2[k])
                {
                    if (k == len - 1 && len1 > len2)
                        return string.Empty;

                    continue;
                }

                adj[word1[k]].Add(word2[k]);
                break;
            }
        }

        var result = new List<char>();
        var visited = new Dictionary<char, bool>(); // False - visited, in the path, True - visited, not in the path
        foreach ((var k, var v) in adj)
        {
            var res = AlienOrderDFS(k, adj, visited, result);
            if (!res)
            {
                return string.Empty;
            }
        }
 
        result.Reverse(); // собрали результат в обратном порядке, т.к. мы добавляем символы в результат после обработки всех их соседей
        return new string(result.ToArray());
    }

    // TODO: разобраться почему это работает
    private static bool AlienOrderDFS(char c, Dictionary<char, HashSet<char>> adj, Dictionary<char, bool> visited, List<char> result)
    {
        if (visited.ContainsKey(c)) // уже обрабатывается либо в текущем пути
            return visited[c]; // если true, значит уже обработались, если false, значит мы в текущем пути, нашли цикл

        visited[c] = false; // отмечаем, что символ у нас как бы в текушем пути
        foreach (var c1 in adj[c])
        {
            var res = AlienOrderDFS(c1, adj, visited, result);
            if (!res)
            {
                return false; // нашли цикл где-то в глубине, значит нет решения
            }
        }

        visited[c] = true; // убираем из текущего пути, отмечаем, что символ обработан
        result.Add(c); // добавляем символ в результат, он будет в конце, т.к. мы добавляем его после обработки всех его соседей
        return true; // все ОК
    }
    #endregion
    // #15
    // 271. Encode and Decode Strings
    // Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.
    #region 271. Encode and Decode Strings
    public class Codec
    {
        // Encodes a list of strings to a single string.
        public string encode(IList<string> strs)
        {
            var builder = new StringBuilder();
            foreach (var str in strs)
            {
                builder.Append(str.Length);
                builder.Append(':');
                builder.Append(str);
            }
            return builder.ToString();
        }

        // Decodes a single string to a list of strings.
        public IList<string> decode(string s)
        {
            var n = s.Length;
            var result = new List<string>();
            var i = 0;
            while (i < n)
            {
                var length = 0;
                while (i < n && s[i] != ':')
                {
                    var d = (int)(s[i] - '0');
                    length = length * 10 + d;
                    i++;
                }
                i++;
                result.Add(s.Substring(i, length));
                i += length;
            }

            return result;
        }
    }

    // Не универсальное решение, т.к. не работает для строк, которые содержат '\n', но для тестов на LeetCode работает
    public class CodecGimmick
    {
        // Encodes a list of strings to a single string.
        public string encode(IList<string> strs)
        {
            return string.Join('\n', strs.ToArray());
        }

        // Decodes a single string to a list of strings.
        public IList<string> decode(string s)
        {
            return s.Split('\n');
        }
    }
    #endregion
    // #16
    // 19. Remove Nth Node From End of List
    // Given the head of a linked list, remove the nth node from the end of the list and return its head.
    // List, Two Pointers
    public static ListNode RemoveNthFromEnd(ListNode head, int n)
    {
        var firstPointer = head;
        ListNode secondPointer = null;

        var count = 0;
        while (firstPointer is not null)
        {
            firstPointer = firstPointer.next;
            count++;

            if (secondPointer is not null)
            {
                secondPointer = secondPointer.next;
            }
            if (count == n + 1)
            {
                secondPointer = head;
            }
        }

        if (secondPointer is not null)
        {
            secondPointer.next = secondPointer.next.next;
        }
        else if (n == count)
        {
            return head.next; // Если нужно удалить первый элемент
        }

        return head;
    }

    // #17
    // 20. Valid Parentheses
    // Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    // Stack
    public static bool IsValid(string s)
    {
        var opens = new Stack<char>();
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(' || s[i] == '{' || s[i] == '[')
            {
                opens.Push(s[i]);
                continue;
            }

            if (s[i] == ')' || s[i] == '}' || s[i] == ']')
            {
                if (opens.Count <= 0)
                {
                    return false;
                }

                var open = opens.Pop();

                if (open == '(' && s[i] != ')')
                {
                    return false;
                }
                if (open == '{' && s[i] != '}')
                {
                    return false;
                }
                if (open == '[' && s[i] != ']')
                {
                    return false;
                }
            }
        }

        if (opens.Count > 0)
        {
            return false;
        }

        return true;
    }

    // #18
    // 21. Merge Two Sorted Lists
    // You are given the heads of two sorted linked lists list1 and list2.
    // Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
    public static ListNode MergeTwoLists(ListNode list1, ListNode list2)
    {
        if (list1 is null && list2 is null)
        {
            return null;
        }
        if (list1 is null)
        {
            return list2;
        }
        if (list2 is null)
        {
            return list1;
        }

        var result = list1.val < list2.val
            ? list1
            : list2;
        var curr = new ListNode();
        while (list1 is not null && list2 is not null)
        {
            if (list1.val < list2.val)
            {
                curr.next = list1;
                list1 = list1.next;
            }
            else
            {
                curr.next = list2;
                list2 = list2.next;
            }
            curr = curr.next;
        }
        if (list1 is null)
        {
            curr.next = list2;
        }
        if (list2 is null)
        {
            curr.next = list1;
        }
        return result;
    }

    // #19
    // 23. Merge k Sorted Lists
    // You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    // Merge all the linked-lists into one sorted linked-list and return it.
    // Идея: использовать подход "разделяй и властвуй", рекурсивно сливая попарно списки
    // Binary Divide and Conquer, Linked List, Merge Sort
    // O(n log k), где n - общее количество элементов во всех списках, k - количество списков

    #region 23. Merge k Sorted Lists
    public static ListNode MergeKLists(ListNode[] lists)
    {
        var n = lists.Length;
        if (n == 0) return null;
        if (n == 1) return lists[0];

        return MergeKLists(lists, 0, lists.Length - 1);
    }

    private static ListNode MergeKLists(ListNode[] lists, int l, int r)
    {
        if (l == r)
        {
            return lists[l];
        }

        ListNode list1 = null;
        ListNode list2 = null;
        if (l == r - 1)
        {
            list1 = lists[l];
            list2 = lists[r];
        }
        else
        {
            var m = l + (r - l) / 2;
            list1 = MergeKLists(lists, l, m);
            list2 = MergeKLists(lists, m + 1, r);
        }

        return MergeTwoListsForMergeKLists(list1, list2);
    }

    private static ListNode MergeTwoListsForMergeKLists(ListNode list1, ListNode list2)
    {
        if (list1 is null && list2 is null) return null;
        if (list1 is null) return list2;
        if (list2 is null) return list1;

        var dummy = new ListNode();
        var cur = dummy;
        while (list1 is not null && list2 is not null)
        {
            if (list1.val < list2.val)
            {
                cur.next = list1;
                list1 = list1.next;
            }
            else
            {
                cur.next = list2;
                list2 = list2.next;
            }

            cur = cur.next;
        }

        if (list1 is null && list2 is not null)
        {
            cur.next = list2;
        }
        if (list2 is null && list1 is not null)
        {
            cur.next = list1;
        }

        return dummy.next;
    }
    #endregion

    // #20
    // 152. Maximum Product Subarray
    // Given an integer array nums, find a subarray that has the largest product, and return the product.
    // Т.е. вернуть максимальное произведение непрерывной подпоследовательности массива
    // Трюк в том, что из-за отрицательных чисел нужно хранить и минимальное произведение на текущем шаге
    // т.к. в случае отричательного числа минимальное произведение может стать максимальным
    // Kaden's Algorithm but modified to track min product as well
    public static int MaxProduct(int[] nums)
    {
        var result = nums[0];
        var maxP = nums[0];
        var minP = nums[0];
        for (int i = 1; i < nums.Length; i++)
        {
            var tmpMax = nums[i] * maxP;
            var tmpMin = nums[i] * minP;
            maxP = Math.Max(nums[i], Math.Max(tmpMax, tmpMin));
            minP = Math.Min(nums[i], Math.Min(tmpMax, tmpMin));

            result = Math.Max(result, maxP);
        }
        return result;
    }

    // #21
    // 153. Find Minimum in Rotated Sorted Array
    // Suppose an array of length n sorted in ascending order is rotated between 1 and n times.
    // Binary Search
    #region 153. Find Minimum in Rotated Sorted Array
    // O(log n)
    public static int FindMin(int[] nums)
    {
        var l = 0;
        var r = nums.Length - 1;
        while (l < r)
        {
            var m = l + (r - l) / 2;

            if (nums[m] < nums[r])
            {
                r = m;
            }
            else
            {
                l = m + 1;
            }
        }
        return nums[l];
    }

    // O(n)
    public static int FindMinLinearTime(int[] nums)
    {
        for (int i = 0; i < nums.Length - 1; i++)
        {
            if (nums[i] > nums[i + 1])
            {
                return nums[i + 1];
            }
        }
        return nums[0];
    }
    #endregion

    // #22
    // 33. Search in Rotated Sorted Array
    // Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
    // Array, Binary Search
    public static int Search(int[] nums, int target)
    {
        var l = 0;
        var r = nums.Length - 1;

        while (l <= r)
        {
            var m = l + (r - l) / 2;
            if (nums[m] == target)
            {
                return m;
            }

            // Если левая часть отсортирована
            if (nums[l] <= nums[m])
            {
                // Проверяем, лежит ли таргет в отсортированной левой части
                if (nums[l] <= target && nums[m] > target)
                {
                    r = m - 1;
                }
                // Если нет, ищем в правой части
                else
                {
                    l = m + 1;
                }
            }
            // Иначе правая часть отсортирована, все аналогично
            else
            {
                if (nums[r] >= target && nums[m] < target)
                {
                    l = m + 1;
                }
                else
                {
                    r = m - 1;
                }
            }
        }

        return -1;
    }

    // #23
    // 417. Pacific Atlantic Water Flow
    // There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean.
    // The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.
    // The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west
    // if the neighboring cell's height is less than or equal to the current cell's height.
    // Water can flow from any cell adjacent to an ocean into the ocean.
    // Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
    // BFS, O(m * n)
    #region 417. Pacific Atlantic Water Flow
    public static IList<IList<int>> PacificAtlantic(int[][] heights)
    {
        var m = heights.Length;
        var n = heights[0].Length;

        IList<IList<int>> result = new List<IList<int>>();
        var reachP = new bool[m, n];
        var reachA = new bool[m, n];
        var queueP = new Queue<(int, int)>();
        var queueA = new Queue<(int, int)>();
        var dI = new int[] { -1, 1, 0, 0 };
        var dJ = new int[] { 0, 0, -1, 1 };

        for (int i = 0; i < m; i++)
        {
            queueP.Enqueue((i, 0));
            queueA.Enqueue((i, n - 1));
        }

        for (int j = 0; j < n; j++)
        {
            queueP.Enqueue((0, j));
            queueA.Enqueue((m - 1, j));
        }

        while (queueP.Count > 0)
        {
            (int i, int j) = queueP.Dequeue();
            if (reachP[i, j]) continue;
            reachP[i, j] = true;

            for (int k = 0; k < 4; k++)
            {
                var di = i + dI[k];
                var dj = j + dJ[k];
                if (di < 0 || dj < 0 || di >= m || dj >= n) continue;
                if (heights[i][j] > heights[di][dj]) continue;

                queueP.Enqueue((di, dj));
            }
        }

        while (queueA.Count > 0)
        {
            (int i, int j) = queueA.Dequeue();
            if (reachA[i, j]) continue;

            reachA[i, j] = true;
            if (reachP[i, j])
            {
                result.Add(new List<int> { i, j });
            }

            for (int k = 0; k < 4; k++)
            {
                var di = i + dI[k];
                var dj = j + dJ[k];
                if (di < 0 || dj < 0 || di >= m || dj >= n) continue;
                if (heights[i][j] > heights[di][dj]) continue;

                queueA.Enqueue((di, dj));
            }
        }

        return result;
    }
    #endregion

    // #24
    // 39. Combination Sum
    // Given an array of distinct integers candidates and a target integer target,
    // return a list of all unique combinations of candidates where the chosen numbers sum to target.
    // The same number may be chosen from candidates an unlimited number of times.
    // Two combinations are unique if the frequency of at least one of the chosen numbers is different.
    // Backtracking, O(2^n * target)
    #region 39. Combination Sum
    public static IList<IList<int>> CombinationSum(int[] candidates, int target)
    {
        IList<IList<int>> result = new List<IList<int>>();
        CSum(0, 0, new List<int>(), candidates, target, result);

        return result;
    }

    private static void CSum(
        int i,
        int currSum,
        IList<int> comb,
        int[] candidates,
        int target,
        IList<IList<int>> result)
    {
        if (currSum == target)
        {
            var copyComb = new List<int>(comb);
            result.Add(copyComb);
            return;
        }

        if (i >= candidates.Length || currSum > target) return;

        comb.Add(candidates[i]);
        var newSum = currSum + candidates[i];
        CSum(i, currSum + candidates[i], comb, candidates, target, result);

        comb.RemoveAt(comb.Count - 1);
        CSum(i + 1, currSum, comb, candidates, target, result);
    }
    #endregion

    // #25
    // 295. Find Median from Data Stream
    // The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
    // Implement the MedianFinder class:
    // - void addNum(int num) adds the integer num from the data stream to the data structure.
    // - double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
    // implemented in MedianFinder.cs

    // #26
    // 424. Longest Repeating Character Replacement
    // Sliding Window
    // TODO: попробовать другой вариант решения с подсчетом частот символов в окне т.н. "classic solution"
    // O(n * 26) = O(n) - time complexity
    // O(1) - memory complexity
    #region 424. Longest Repeating Character Replacement
    public static int CharacterReplacement(string s, int k)
    {
        var n = s.Length;
        // Ранний выход, если можно тупо заменить все символы кроме одного
        if (k >= n - 1) return n;

        var result = 0;
        for (int i = 0; i < 26; i++)
        {
            var currentLetter = (char)('A' + i);
            var currentMaxLen = 0;
            var replaces = 0;
            var r = 0;
            for (int l = 0; l < n; l++)
            {
                while (r < n && (s[r] == currentLetter || replaces < k))
                {
                    if (s[r] != currentLetter) replaces++;

                    r++;
                }
                currentMaxLen = Math.Max(currentMaxLen, r - l);

                if (s[l] != currentLetter) replaces--;
            }

            result = Math.Max(result, currentMaxLen);

            if (result == n) // Ранний выход, длиннее уже не может быть
                break;
        }

        return result;
    }
    #endregion

    // #27
    // 300. Longest Increasing Subsequence
    // Given an integer array nums, return the length of the longest strictly increasing subsequence.
    // DP
    // O(n^2) time complexity
    // TODO: implement O(n log n) solution with binary search
    #region 300. Longest Increasing Subsequence
    public static int LengthOfLIS(int[] nums)
    {
        var n = nums.Length;
        if (n <= 1) return n;

        var result = 0;
        var cache = new int[n];
        for (int i = n - 1; i >= 0; i--)
        {
            var currMaxLen = 1;
            for (int j = i + 1; j < n; j++)
            {
                if (nums[j] > nums[i])
                {
                    currMaxLen = Math.Max(currMaxLen, 1 + cache[j]);
                }
            }
            cache[i] = currMaxLen;
            result = Math.Max(result, currMaxLen);
        }

        return result;
    }
    #endregion

    // #28
    // 48. Rotate Image
    // You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
    // You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
    // O(n^2) time complexity
    #region 48. Rotate Image
    public static void Rotate(int[][] matrix)
    {
        var n = matrix.Length;
        if (n == 1) return;
        var stepLimit = n / 2;
        for (var i = 0; i < stepLimit; i++)
        {
            for (int j = i; j < n - i - 1; j++)
            {
                // Для понимания, какие элементы меняются местами    ---t-->
                // var t = new int[] { i, j };                      ^       | 
                // var l = new int[] { n - 1 - j, i };             l|       |r
                // var r = new int[] { j, n - 1 - i };              |       v
                // var b = new int[] { n - 1 - i, n - 1 - j };       <--b---

                var tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = tmp;
            }
        }
    }
    #endregion

    // #29
    // 49. Group Anagrams
    // Given an array of strings strs, group the anagrams together. You can return the answer in any order.
    // Frequency Count as Hash Key
    #region 49. Group Anagrams
    public static IList<IList<string>> GroupAnagrams(string[] strs)
    {
        if (strs is null || strs.Length == 0)
        {
            return new List<IList<string>>();
        }

        var n = strs.Length;
        var groups = new Dictionary<string, List<string>>();
        for (int i = 0; i < n; i++)
        {
            var keyArr = new int[26];
            for (int j = 0; j < strs[i].Length; j++)
            {
                var index = (int)(strs[i][j] - 'a');
                keyArr[index]++;
            }
            var key = string.Join(",", keyArr);
            if (!groups.ContainsKey(key))
            {
                groups[key] = new List<string>();
            }
            groups[key].Add(strs[i]);
        }

        var result = new List<IList<string>>();
        foreach (var item in groups)
        {
            var group = new List<string>();
            foreach (var str in item.Value)
            {
                group.Add(str);
            }
            result.Add(group);
        }

        return result;
    }
    #endregion

    // #30
    // 435. Non-overlapping Intervals
    // Given an array of intervals intervals where intervals[i] = [starti, endi],
    // return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
    // Note that intervals which only touch at a point are non-overlapping. For example, [1, 2] and [2, 3] are non-overlapping.
    // Greedy, Sort
    // 0(n log n) time complexity because of sorting
    #region 435. Non-overlapping Intervals

    // Option 1: sort by start time
    public static int EraseOverlapIntervals(int[][] intervals)
    {
        var result = 0;

        // sort by interval start time
        // var sortedIntervals = intervals
        //     .OrderBy(x => x[0])
        //     .ToArray();

        // sort by interval start time
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

        var currentEndTime = int.MinValue;
        foreach (var interval in intervals)
        {
            if (currentEndTime <= interval[0])
            {
                currentEndTime = interval[1];
            }
            else
            {
                result++;
                currentEndTime = Math.Min(interval[1], currentEndTime);
            }
        }

        return result;
    }

    // Option 2: sort by end time
    public static int EraseOverlapIntervals1(int[][] intervals)
    {
        var result = 0;
        var n = intervals.Length;
        // sort by interval end time
        Array.Sort(intervals, (a, b) => a[1].CompareTo(b[1]));

        var prevTime = int.MinValue;
        foreach (var interval in intervals)
        {
            if (prevTime <= interval[0])
            {
                prevTime = interval[1];
            }
            else
            {
                result++;
            }
        }

        return result;
    }
    #endregion

    // #31
    // 53. Maximum Subarray
    // Given an integer array nums, find the subarray with the largest sum, and return its sum.
    // Kadane’s Algorithm
    // O(n) time complexity
    #region 53. Maximum Subarray
    public static int MaxSubArray(int[] nums)
    {
        var maxSum = nums[0];
        var currentSum = nums[0];
        for (var i = 1; i < nums.Length; i++)
        {
            currentSum = Math.Max(nums[i], currentSum + nums[i]);
            maxSum = Math.Max(currentSum, maxSum);
        }

        return maxSum;
    }
    #endregion

    // #32
    // 54. Spiral Matrix
    // Given an m x n matrix, return all elements of the matrix in spiral order.
    // O(m * n) time complexity
    #region 54. Spiral Matrix
    public static IList<int> SpiralOrder_usingStep(int[][] matrix)
    {
        var result = new List<int>();
        if (matrix.Length == 0) return result;
        if (matrix[0].Length == 0) return result;
        var m = matrix.Length;
        var n = matrix[0].Length;
        var minMN = Math.Min(m, n);
        var stepLimit = minMN / 2 + minMN % 2;
        for (int step = 0; step < stepLimit; step++)
        {
            // top
            for (int j = step; j < n - step; j++)
            {
                result.Add(matrix[step][j]);
            }
            // right
            for (int i = step + 1; i < m - step; i++)
            {
                result.Add(matrix[i][n - step - 1]);
            }
            // bottom
            for (int j = n - step - 2; j > step && (m - step - 1) > step; j--)
            {
                result.Add(matrix[m - step - 1][j]);
            }
            // left
            for (int i = m - step - 1; i > step && (n - step - 1) > step; i--)
            {
                result.Add(matrix[i][step]);
            }
        }

        return result;
    }

    // 54. Spiral Matrix using boundaries: top left right bottom
    // Makes it more readable
    public static IList<int> SpiralOrder(int[][] matrix)
    {
        var result = new List<int>();
        if (matrix.Length == 0) return result;
        if (matrix[0].Length == 0) return result;
        var m = matrix.Length;
        var n = matrix[0].Length;
        var top = 0;
        var left = 0;
        var right = n - 1;
        var bottom = m - 1;
        while (top <= bottom && left <= right)
        {
            // top
            for (int j = left; j <= right; j++)
            {
                result.Add(matrix[top][j]);
            }
            top++;

            // right
            for (int i = top; i <= bottom; i++)
            {
                result.Add(matrix[i][right]);
            }
            right--;

            // bottom
            if (top <= bottom)
            {
                for (int j = right; j >= left; j--)
                {
                    result.Add(matrix[bottom][j]);
                }
                bottom--;
            }
            // left
            if (left <= right)
            {
                for (int i = bottom; i >= top; i--)
                {
                    result.Add(matrix[i][left]);
                }
                left++;
            }
        }

        return result;
    }

    #endregion

    // #33
    // 55. Jump Game
    // You are given an integer array nums. You are initially positioned at the array's first index,
    // and each element in the array represents your maximum jump length at that position.
    // Return true if you can reach the last index, or false otherwise.
    #region 55. Jump Game

    // Greedy
    // O(n) time complexity
    public static bool CanJump(int[] nums)
    {
        var n = nums.Length;
        if (n <= 1) return true;

        var maxIndex = 0;

        for (int i = 0; i <= maxIndex; i++)
        {
            maxIndex = Math.Max(maxIndex, i + nums[i]);
            if (maxIndex >= n - 1) return true;
        }

        return false;
    }

    // DP + Memoization
    // O(n^2) time complexity
    // NOT OPTIMAL
    public static bool CanJumpDP(int[] nums)
    {
        var n = nums.Length;
        return CanJumpDP(nums, 0, n, new Dictionary<int, bool>());
    }

    private static bool CanJumpDP(int[] nums, int i, int n, Dictionary<int, bool> cache)
    {
        if (cache.ContainsKey(i)) return cache[i];

        if (i >= n - 1 || i + nums[i] >= n - 1)
        {
            cache[i] = true;
            return true;
        }

        var result = false;
        var lastIndex = Math.Min(n - 1, i + nums[i]);
        for (int j = i + 1; j <= lastIndex; j++)
        {
            result = result || CanJumpDP(nums, j, n, cache);
        }

        cache[i] = result;
        return result;
    }
    #endregion

    // #34
    // 56. Merge Intervals
    // Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals,
    // and return an array of the non-overlapping intervals that cover all the intervals in the input.
    // Greedy, Sort
    // O(n log n) time complexity because of sorting
    #region 56. Merge Intervals
    public static int[][] Merge(int[][] intervals)
    {
        var n = intervals.Length;
        if (n <= 1) return intervals;

        var result = new List<int[]>();
        
        // sort by interval start time
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

        var prevInterval = intervals[0]; // as an option, create a copy to avoid mutating input array
        for (int i = 1; i < n; i++)
        {
            if (prevInterval[1] < intervals[i][0])
            {
                result.Add(prevInterval);
                prevInterval = intervals[i]; // as an option, create a copy to avoid mutating input array
            }
            else
            {
                prevInterval[1] = Math.Max(prevInterval[1], intervals[i][1]);
            }
        }
        result.Add(prevInterval);
        return result.ToArray();
    }
    #endregion

    // #35
    // 57. Insert Interval
    // You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi]
    // represent the start and the end of the i-th interval and intervals is sorted in ascending order by starti.
    // You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
    // Insert newInterval into intervals such that intervals is still sorted in ascending order by starti
    // and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
    // Return intervals after the insertion.
    #region 57. Insert Interval
    // First attempt: one loop
    public static int[][] InsertOneLoop(int[][] intervals, int[] newInterval)
    {
        var n = intervals.Length;
        if (n == 0)
        {
            return new[] { newInterval };
        }

        var result = new List<int[]>();
        var inserted = false;
        for (int i = 0; i < n; i++)
        {
            if (inserted)
            {
                result.Add(intervals[i]);
                continue;
            }

            if (intervals[i][1] < newInterval[0])
            {
                result.Add(intervals[i]);
                continue;
            }

            if (intervals[i][0] > newInterval[1])
            {
                inserted = true;
                result.Add(newInterval);
                result.Add(intervals[i]);
                continue;
            }

            // overlapping
            if (intervals[i][0] <= newInterval[1] && intervals[i][1] >= newInterval[0])
            {
                newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
                newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
            }
        }

        if (!inserted)
        {
            result.Add(newInterval);
        }

        return result.ToArray();
    }

    // Second attempt: three loops, more readable
    public static int[][] Insert(int[][] intervals, int[] newInterval)
    {
        var n = intervals.Length;
        if (n == 0)
        {
            return new[] { newInterval };
        }

        var result = new List<int[]>();
        var i = 0;
        // add all intervals that are strictly before
        while (i < n && intervals[i][1] < newInterval[0])
        {
            result.Add(intervals[i]);
            i++;
        }

        // merge if there are any overlapping intervals
        while (i < n && intervals[i][0] <= newInterval[1] && intervals[i][1] >= newInterval[0])
        {
            newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.Add(newInterval);

        // add all intervals that are strictly after
        while (i < n)
        {
            result.Add(intervals[i]);
            i++;
        }

        return result.ToArray();
    }
    #endregion
    
    // #36
    // 572. Subtree of Another Tree
    // Subtree of Another Tree
    #region Subtree of Another Tree
    public static bool IsSubtree(TreeNode root, TreeNode subRoot)
    {
        return IsSubSubtree(root, subRoot, false);
    }

    private static bool IsSubSubtree(TreeNode root, TreeNode subRoot, bool strict)
    {
        if (subRoot is null && root is null)
        {
            return true;
        }

        if (subRoot is null || root is null)
        {
            return false;
        }

        if (root.val == subRoot.val)
        {
            var isSubtree = IsSubSubtree(root.left, subRoot.left, true)
                && IsSubSubtree(root.right, subRoot.right, true);
            if (isSubtree)
                return true;
        }

        if (!strict)
        {
            return IsSubSubtree(root.left, subRoot, false)
                || IsSubSubtree(root.right, subRoot, false);
        }

        return false;
    }
    #endregion

    // #37
    // 62. Unique Paths
    // There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]).
    // The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
    // Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
    // O(m * n) time complexity
    //
    // Важно: Можно решить через формулу C(m+n-2, m-1) или C(m+n-2, n-1), например (m + n - 2)! / ((m - 1)! * (n - 1)!) 
    // При этом важно следить, чтобы не было переполнения при вычислении факториала
    // Так что возможно DP тут предпочтительнее
    #region 62. Unique Paths

    // First attempt: DP + Memoization, recursive
    public static int UniquePathsR(int m, int n, Dictionary<(int, int), int> cache = null)
    {
        if (n <= 1 || m <= 1) return 1;

        if (n == 2) return m;
        if (m == 2) return n;

        var key = (Math.Min(m, n), Math.Max(m, n));
        if (cache is null)
        {
            cache = new Dictionary<(int, int), int>();
        }
        else
        {
            if (cache.ContainsKey(key))
            {
                return cache[key];
            }
        }

        var result = UniquePathsR(m - 1, n, cache) + UniquePathsR(m, n - 1, cache);
        cache[key] = result;
        return result;
    }

    // Second attempt: DP + Memoization, iterative
    public static int UniquePaths(int m, int n)
    {
        if (n <= 1 || m <= 1) return 1;

        if (n == 2) return m;
        if (m == 2) return n;

        var cache = new int[m, n];

        for (int i = 0; i < m; i++) { cache[i, 0] = 1; }
        for (int j = 1; j < n; j++) { cache[0, j] = 1; }

        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                cache[i, j] = cache[i - 1, j] + cache[i, j - 1];
            }
        }

        return cache[m - 1, n - 1];
    }

    #endregion

    // #38
    // 190. Reverse Bits
    // Reverse bits of a given 32 bits signed integer.
    // Bit Manipulation
    // O(n) time complexity where n is number of bits (32) in fact it's O(1)
    #region 190. Reverse Bits
    public static int ReverseBits(int n)
    {
        var result = 0;
        for (int i = 0; i < 32; i++)
        {
            if (((1 << i) & n) > 0)
            {
                result |= (1 << (31 - i));
            }
        }
        return result;
    }
    #endregion

    // #39
    // 191. Number of 1 Bits
    // Given a positive integer n, write a function that returns the number of set bits in its binary representation (also known as the Hamming weight).
    // Bit Manipulation
    // O(n) time complexity where n is number of bits (32) in fact it's O(1)
    #region 191. Number of 1 Bits
    // My solution
    public static int HammingWeight(int n)
    {
        var result = 0;
        for (int i = 0; i < 32; i++)
        {
            if (((1 << i) & n) > 0)
            {
                result++;
            }
        }

        return result;
    }

    // Bit Manipulation Trick
    // Идея: n & (n - 1) сбрасывает (устанавливает в 0) самый младший установленный бит в n
    // n       = 1011000
    // n - 1   = 1010111
    // n&(n-1) = 1010000
    public static int HammingWeightTrick(int n)
    {
        int count = 0;
        while (n != 0)
        {
            n &= (n - 1); 
            count++;
        }
        return count;
    }
    #endregion

    // #40
    // 449. Serialize and Deserialize BST
    // Serialization is converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer,
    // or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
    // Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work.
    // You need to ensure that a binary search tree can be serialized to a string, and this string can be deserialized to the original tree structure.
    // The encoded string should be as compact as possible.
    // implemented in Codec.cs

    // #41
    // 322. Coin Change
    // You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
    // Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
    // You may assume that you have an infinite number of each kind of coin.
    // DP
    #region 322. Coin Change
    // 2D DP
    // NOT OPTIMAL FOR MEMORY
    public static int CoinChange2D_DP(int[] coins, int amount)
    {
        if (amount == 0) return 0;
        var n = coins.Length;
        if (n == 0) return -1;

        var dp = new int[amount, n];

        for (int a = 0; a < amount; a++)
        {
            for (int c = 0; c < n; c++)
            {
                if (coins[c] > a + 1)
                {
                    dp[a, c] = c > 0
                        ? dp[a, c - 1]
                        : int.MaxValue;
                }
                else if (coins[c] == a + 1)
                {
                    dp[a, c] = 1;
                }
                else // coins[c] < a + 1
                {
                    var prevAmountPlusCoin = dp[a - coins[c], c] < int.MaxValue
                        ? dp[a - coins[c], c] + 1
                        : int.MaxValue;
                    var sameAmountNoCoin = c > 0
                        ? dp[a, c - 1]
                        : int.MaxValue;

                    dp[a, c] = Math.Min(prevAmountPlusCoin, sameAmountNoCoin);
                }

            }
        }

        return dp[amount - 1, n - 1] < int.MaxValue
            ? dp[amount - 1, n - 1]
            : -1;
    }

    // 1D DP
    public static int CoinChange(int[] coins, int amount)
    {
        if (amount == 0) return 0;
        var n = coins.Length;
        if (n == 0) return -1;
        var maxValue = int.MaxValue / 2;

        var dp = new int[amount];
        for (int a = 0; a < amount; a++)
        {
            dp[a] = maxValue;
        }

        for (int a = 0; a < amount; a++)
        {
            for (int c = 0; c < n; c++)
            {
                if (coins[c] == a + 1)
                {
                    dp[a] = 1;
                }
                else if (coins[c] < a + 1)
                {
                    dp[a] = Math.Min(dp[a - coins[c]] + 1, dp[a]);
                }
            }
        }

        return dp[amount - 1] < maxValue
            ? dp[amount - 1]
            : -1;
    }
    #endregion

    // #42
    // 323. Number of Connected Components in an Undirected Graph
    // You have a graph of n nodes. You are given an integer n and an array edges
    // where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.
    // Return the number of connected components in the graph.
    // DFS
    // O(n + m) time complexity where n is number of nodes and m is number of edges
    #region 323. Number of Connected Components in an Undirected Graph
    public static int CountComponents(int n, int[][] edges)
    {
        var adj = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new List<int>();
        }

        for (int i = 0; i < edges.Length; i++)
        {
            adj[edges[i][0]].Add(edges[i][1]);
            adj[edges[i][1]].Add(edges[i][0]);
        }

        var result = 0;
        var visited = new bool[n];
        var stack = new Stack<int>();
        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;
            stack.Push(i);
            result++;
            while (stack.Count > 0)
            {
                var j = stack.Pop();
                if (visited[j]) continue;
                visited[j] = true;
                foreach (var k in adj[j])
                {
                    stack.Push(k);
                }
            }
        }

        return result;
    }
    #endregion

    // #43
    // 70. Climbing Stairs
    // You are climbing a staircase. It takes n steps to reach the top.
    // Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    // Fibonacci-like DP
    // O(n) time complexity
    #region 70. Climbing Stairs
    public static int ClimbStairs(int n)
    {
        if (n <= 3) return n;

        var prev1 = 3;
        var prev2 = 2;
        var result = 0;
        for (int i = 4; i <= n; i++)
        {
            result = prev2 + prev1;
            prev2 = prev1;
            prev1 = result;
        }
        return result;
    }
    #endregion

    // #44
    // 198. House Robber
    // You are a professional robber planning to rob houses along a street.
    // Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them
    // is that adjacent (соседние) houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
    // Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
    // DP + Memoization
    #region 198. House Robber, recursive
    // O(n) time complexity
    // O(n) space complexity
    public static int RobRecursive(int[] nums)
    {
        return RobRecursive(nums, 0, new Dictionary<int, int>());
    }

    private static int RobRecursive(int[] nums, int i, Dictionary<int, int> cache)
    {
        var n = nums.Length;
        if (i >= n) return 0;
        if (i == n - 1) return nums[n - 1];
        if (cache.ContainsKey(i)) return cache[i];

        var result = Math.Max(RobRecursive(nums, i + 1, cache), nums[i] + RobRecursive(nums, i + 2, cache));
        cache[i] = result;
        return result;
    }
    #endregion
    
    #region 198. House Robber, iterative
    // O(n) time complexity
    // O(1) space complexity
    public static int Rob(int[] nums)
    {
        var result = 0;
        var prevEx = 0;
        var prevIn = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            result = Math.Max(nums[i] + prevIn, prevEx);
            prevIn = prevEx;
            prevEx = result;
        }

        return result;
    }
    #endregion

    // #45
    // 200. Number of Islands
    // Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
    // An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
    // You may assume all four edges of the grid are all surrounded by water.
    // DFS
    // O(m * n) time complexity
    // O(m * n) space complexity because of stack
    // Идея: находим '1', увеличиваем счетчик островов, заливаем весь остров (DFS) меняя '1' на 'i', чтобы не считать его повторно

    #region 200. Number of Islands

    public static int NumIslands(char[][] grid)
    {
        var m = grid.Length;
        var n = grid[0].Length;
        var dI = new[] { -1, 1, 0, 0 };
        var dJ = new[] { 0, 0, -1, 1 };
        var result = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (grid[i][j] == '1')
                {
                    result++;

                    var stack = new Stack<(int, int)>();
                    grid[i][j] = 'i';
                    stack.Push((i, j));
                    while (stack.Count > 0)
                    {
                        (int ii, int jj) = stack.Pop();
                        for (int k = 0; k < 4; k++)
                        {
                            var iid = ii + dI[k];
                            var jjd = jj + dJ[k];

                            if (iid >= 0 && iid < m && jjd >= 0 && jjd < n && grid[iid][jjd] == '1')
                            {
                                grid[iid][jjd] = 'i';
                                stack.Push((iid, jjd));
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    #endregion

    // #46
    // 73. Set Matrix Zeroes
    // Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    // You must do it in place.
    // O(m * n) time complexity
    #region 73. Set Matrix Zeroes

    // First attempt: O(m + n) space complexity
    public static void SetZeroesOmn(int[][] matrix)
    {
        var m = matrix.Length;
        var n = matrix[0].Length;

        var rows = new bool[m];
        var cols = new bool[n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (matrix[i][j] == 0)
                {
                    rows[i] = true;
                    cols[j] = true;
                }
            }
        }

        for (int i = 0; i < m; i++)
        {
            if (rows[i])
            {
                for (int j = 0; j < n; j++)
                {
                    matrix[i][j] = 0;
                }
            }
        }

        for (int j = 0; j < n; j++)
        {
            if (cols[j])
            {
                for (int i = 0; i < m; i++)
                {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    // Second attempt: O(1) space complexity
    // Идея: использовать нулевую строку и нулевой столбец для хранения информации о том, какие строки и столбцы нужно обнулить
    // Важно: отдельно хранить флаг x0 для нулевой строки, так как matrix[0][0] кодирует нулевой столбец
    public static void SetZeroes(int[][] matrix)
    {
        var m = matrix.Length;
        var n = matrix[0].Length;
        var x0 = false;
        // Отдельно обрабатываем нулевую строку
        for (int j = 0; j < n; j++)
        {
            if (matrix[0][j] == 0)
            {
                x0 = true;
                break;
            }
        }
        // Заполнияем информацию о том, какие строки и столбцы нужно обнулить
        for (int i = 1; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (matrix[i][j] == 0)
                {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        // Обнуляем строки, начиная с первой (нулевую обработаем отдельно, чтоб не запортачить флаги для обнуления столбцов)
        for (int i = 1; i < m; i++)
        {
            if (matrix[i][0] == 0)
            {
                for (int j = 1; j < n; j++)
                {
                    matrix[i][j] = 0;
                }
            }
        }
        // Обнуляем столбцы
        for (int j = 0; j < n; j++)
        {
            if (matrix[0][j] == 0)
            {
                for (int i = 1; i < m; i++)
                {
                    matrix[i][j] = 0;
                }
            }
        }
        // Обнуляем нулевую строку, если нужно
        if (x0)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[0][j] = 0;
            }
        }
    }

    #endregion

    // #47
    // 76. Minimum Window Substring
    // HARD
    // Given two strings s and t of lengths m and n respectively, return the minimum window substring of s
    // such that every character in t (including duplicates) is included in the window.
    // If there is no such substring, return the empty string "".
    // Sliding Window + HashMap
    // O(m + n) time complexity
    // Идея: Используем скользящее окно. Основная проблема - это оптимизация проверки валидности строки в окне
    // Считаем частоты символов в окне (только тех, что есть в t), если частота текущего символа в окне становится СТРОГО равной его частоте в t,
    // то увеличиваем счетчик charsFound, который показывает, сколько уникальных символов из t мы уже нашли в окне в нужном количестве
    // charsNeeded - количество уникальных символов в строке t
    // charsFound - количество верно найденных уникальных символов в текущем окне
    // charsFound обновляется строго когда частота найденного символа в окне становится равной его частоте в строке t
    // Это гарантирует нам, что когда charsFound == charsNeeded, мы действительно нашли окно, содержащее все символы строки t
    #region 76. Minimum Window Substring
    public static string MinWindow(string s, string t)
    {
        var m = s.Length;
        var n = t.Length;
        if (n > m || n == 0 || m == 0) return string.Empty;

        var tFreq = new Dictionary<char, int>();
        var sFreq = new Dictionary<char, int>();
        var charsFound = 0;
        var charsNeeded = 0;
        for (int i = 0; i < n; i++)
        {
            if (!tFreq.ContainsKey(t[i]))
            {
                tFreq[t[i]] = 1;
                charsNeeded++;
            }
            else
            {
                tFreq[t[i]]++;
            }
            sFreq[t[i]] = 0;
        }

        var resultI = 0;
        var resultLen = int.MaxValue;
        var l = 0;

        for (int i = 0; i < m; i++)
        {
            if (tFreq.TryGetValue(s[i], out var tVal))
            {
                sFreq[s[i]]++;
                if (sFreq[s[i]] == tVal)
                {
                    charsFound++;
                }
            }

            while (charsFound == charsNeeded && l <= i)
            {
                var candidateLen = i - l + 1;
                if (candidateLen < resultLen)
                {
                    resultI = l;
                    resultLen = candidateLen;
                }

                if (tFreq.TryGetValue(s[l], out var tlVal))
                {
                    sFreq[s[l]]--;
                    if (sFreq[s[l]] < tlVal)
                    {
                        charsFound--;
                    }
                }

                l++;
            }
        }

        if (resultLen < int.MaxValue)
            return s.Substring(resultI, resultLen);

        return string.Empty;
    }
    #endregion

    // #48
    // 206. Reverse Linked List
    // Given the head of a singly linked list, reverse the list, and return the reversed list.
    // O(n) time complexity
    // Идея: вставка за временную голову
    #region 206. Reverse Linked List
    public static ListNode ReverseList(ListNode head)
    {
        var tmpHead = new ListNode();

        while (head is not null)
        {
            var h = head.next;
            head.next = tmpHead.next;
            tmpHead.next = head;
            head = h;
        }

        return tmpHead.next;
    }
    #endregion

    // #49
    // 79. Word Search
    // Given an m x n grid of characters board and a string word, return true if word exists in the grid.
    // The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.
    // The same letter cell may not be used more than once.
    // DFS + Backtracking
    // O(m * n * 4^L) time complexity, where L is the length of the word
    #region 79. Word Search

    public static bool Exist(char[][] board, string word)
    {
        var m = board.Length;
        var n = board[0].Length;
        var visited = new bool[m, n];
        var di = new int[] { -1, 1, 0, 0 };
        var dj = new int[] { 0, 0, -1, 1 };
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (WordExists(i, j, 0, board, word, visited, di, dj))
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool WordExists(int i, int j, int c, char[][] board, string word, bool[,] visited, int[] dirI, int[] dirJ)
    {
        if (c >= word.Length) return true;
        if (i < 0 || i >= board.Length || j < 0 || j >= board[0].Length) return false;
        if (visited[i, j]) return false;
        if (board[i][j] != word[c]) return false;
        visited[i, j] = true;

        for (int d = 0; d < 4; d++)
        {
            var di = i + dirI[d];
            var dj = j + dirJ[d];
            if (WordExists(di, dj, c + 1, board, word, visited, dirI, dirJ))
            {
                return true;
            }
        }
        // Важный момент бэктрекинга - если этот путь не подходит, снимаем отметку посещения, чтобы не пересекаться с другими путями
        visited[i, j] = false; 
        return false;
    }

    #endregion

    // #50
    // 207. Course Schedule
    // There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
    // You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
    // For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    // Return true if you can finish all courses. Otherwise, return false.
    // DFS
    // TODO: реализовать BFS
    #region 207. Course Schedule

    public static bool CanFinish(int numCourses, int[][] prerequisites)
    {
        var conn = new List<int>[numCourses];
        for (int i = 0; i < numCourses; i++)
        {
            conn[i] = new List<int>();
        }

        foreach (var edge in prerequisites)
        {
            conn[edge[0]].Add(edge[1]);
        }

        var canFinish = new int[numCourses]; // 0 - not processed, 1 - visited, 2 - can be finished
        for (int i = 0; i < numCourses; i++)
        {
            if (!CanFinish(i, conn, canFinish))
            {
                return false;
            }
        }

        return true;
    }

    private static bool CanFinish(int i, List<int>[] conn, int[] canFinish)
    {
        if (canFinish[i] == 2) return true;
        if (canFinish[i] == 1)
        {
            return false;
        }

        canFinish[i] = 1;
        foreach (int j in conn[i])
        {
            if (CanFinish(j, conn, canFinish) == false)
            {
                return false;
            }
        }

        canFinish[i] = 2;
        return true;
    }
    #endregion

    // #51
    // 208. Implement Trie (Prefix Tree)
    // A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings.
    // There are various applications of this data structure, such as autocomplete and spellchecker.
    // Implement the Trie class:
    // - Trie() Initializes the trie object.
    // - void insert(String word) Inserts the string word into the trie.
    // - boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    // - boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
    // implemented in Trie.cs

    // #52
    // 338. Counting Bits
    // Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
    // DP
    // O(n) time complexity
    // Идея: вычитать из числа i степень двойки pow2 (максимальная степень, которая меньше i).
    // Фактически мы убираем старшую единицу, оставшееся число уже должно быть посчитано.
    // Соответственно берем result[i] = 1 + result[i - pow2]. По ходу обновляем pow2.
    #region 338. Counting Bits

    public static int[] CountBits(int n)
    {
        var result = new int[n + 1];
        var pow2 = 1;
        var nextPow2 = 2;
        for (int i = 1; i <= n; i++)
        {
            if (i == nextPow2)
            {
                result[i] = 1;
                pow2 = nextPow2;
                nextPow2 *= 2;
            }
            else
            {
                result[i] = 1 + result[i - pow2];
            }
        }

        return result;
    }
    
    #endregion

    // #53
    // 211. Design Add and Search Words Data Structure
    // Design a data structure that supports adding new words and finding if a string matches any previously added string.
    // Implement the WordDictionary class:
    // - WordDictionary() Initializes the object.
    // - void addWord(word) Adds word to the data structure, it can be matched later.
    // - bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise.
    //   word may contain dots '.' where dots can be matched with any letter.
    // implemented in WordDictionary.cs

    // #54
    // 212. Word Search II
    // HARD
    // Given an m x n board of characters and a list of strings words, return all words on the board.
    // Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.
    // The same letter cell may not be used more than once in a word.
    // DFS + Trie
    // Идея: построить Trie для слов из списка words, затем для каждой ячейки на доске запустить DFS
    // В процессе DFS мы будем спускаться по Trie, если текущая ячейка соответствует символу в Trie, то продолжаем спускаться,
    // если достигли конца слова в Trie, то добавляем его в результат и удаляем из Trie, чтобы не находить его повторно
    //
    // O(m * n * 3^L) time complexity, где L - средняя длина слова в списке words, так как в худшем случае мы можем проверить все возможные пути на доске для каждого слова
    // 3^L, а не 4^L, потому что назад нельзя, board[i][j] = '#' (visited) зарежет путь назад
    #region 212. Word Search II
    public static IList<string> FindWords(char[][] board, string[] words)
    {
        var m = board.Length;
        var n = board[0].Length;
        var trie = new Trie212();
        var result = new List<string>();

        var dirI = new[] { -1, 1, 0, 0 };
        var dirJ = new[] { 0, 0, -1, 1 };

        for (int i = 0; i < words.Length; i++)
        {
            trie.Insert(words[i]);
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var node = trie.GetNode(board[i][j]);
                if (node is null) continue;
                DFS(i, j, board, result, node, m, n, dirI, dirJ);
            }
        }
        return result;
    }

    private static void DFS(
        int i, int j,
        char[][] board,
        IList<string> result,
        Trie212 node,
        int m, int n, int[] dirI, int[] dirJ)
    {
        var ch = board[i][j];
        board[i][j] = '#'; // visited

        if (node.IsWordEnd)
        {
            result.Add(node.Word);
            node.RemoveWord();
        }

        for (int d = 0; d < 4; d++)
        {
            var di = i + dirI[d];
            var dj = j + dirJ[d];
            if (di < 0 || dj < 0 || di >= m || dj >= n)
                continue;

            if (board[di][dj] == '#') // visited, skip this
                continue;

            var nextNode = node.GetNode(board[di][dj]);
            if (nextNode is null)
                continue;

            DFS(di, dj, board, result, nextNode, m, n, dirI, dirJ);
        }

        board[i][j] = ch; // restore, not visited anymore
    }

    public class Trie212
    {
        private bool _isWordEnd;
        private Trie212[] _nodes;
        private Trie212 _parent;
        private int _index;
        private string _word = string.Empty;
        private int _childrenCount = 0;

        public string Word { get { return _word; } }
        public bool IsWordEnd { get { return _isWordEnd; } }
        public int ChildrenCount { get { return _childrenCount; } }

        public Trie212()
        {
            _nodes = new Trie212[26];
        }

        public void Insert(string word)
        {
            var n = word.Length;
            var current = this;
            for (int i = 0; i < n; i++)
            {
                var index = GetIndex(word[i]);
                if (current._nodes[index] is null)
                {
                    current._nodes[index] = new Trie212();
                    current._childrenCount++;
                    current._nodes[index]._parent = current;
                    current._nodes[index]._index = index;
                }
                if (i == n - 1)
                {
                    current._nodes[index]._isWordEnd = true;
                    current._nodes[index]._word = word;
                }
                current = current._nodes[index];
            }
        }

        public void RemoveWord()
        {
            if (!_isWordEnd)
                return;

            _isWordEnd = false;

            if (_childrenCount == 0 && _parent is not null)
            {
                _parent.RemoveAt(_index);
            }
        }

        private void RemoveAt(int index)
        {
            if (index < 0 || index >= 26) return;

            if (_nodes[index] is not null)
            {
                _nodes[index] = null;
                _childrenCount--;
            }

            if (_childrenCount == 0 && !_isWordEnd && _parent is not null)
            {
                _parent.RemoveAt(_index);
            }
        }

        public Trie212 GetNode(char c)
        {
            return _nodes[GetIndex(c)];
        }

        private int GetIndex(char c) => (int)(c - 'a');
    }
    #endregion

    // #55
    // 213. House Robber II
    // You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed.
    // All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one.
    // Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police
    // if two adjacent houses were broken into on the same night.
    // Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
    // Hint 1: Since House[1] and House[n] are adjacent, they cannot be robbed together. Therefore, the problem becomes to rob either House[1]-House[n-1] or House[2]-House[n],
    // depending on which choice offers more money. Now the problem has degenerated to the House Robber, which is already been solved.
    // DP + Memoization, iterative
    #region 213. House Robber II
    public static int Rob2(int[] nums)
    {
        var n = nums.Length;
        if (n == 1) return nums[0];

        var prevIn = 0;
        var prevEx = 0;
        var result0 = 0;
        // robbing 0th house, but not robbing last house
        for (int i = 0; i < n - 1; i++)
        {
            result0 = Math.Max(prevIn + nums[i], prevEx);
            prevIn = prevEx;
            prevEx = result0;
        }

        prevIn = 0;
        prevEx = 0;
        var resultN = 0;
        // robbing last house, but not robbing 0th house
        for (int i = 1; i < n; i++)
        {
            resultN = Math.Max(prevIn + nums[i], prevEx);
            prevIn = prevEx;
            prevEx = resultN;
        }

        return Math.Max(result0, resultN);
    }
    #endregion

    // #56
    // 217. Contains Duplicate
    // Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
    #region 217. Contains Duplicate
    public static bool ContainsDuplicate(int[] nums)
    {
        var hash = new HashSet<int>();
        var n = nums.Length;

        for (int i = 0; i < n; i++)
        {
            // HashSet.Add method returns false if the item was already present
            if (!hash.Add(nums[i])) return true;
        }

        return false;
    }
    #endregion

    // #57
    // 91. Decode Ways
    // You have intercepted a secret message encoded as a string of numbers. The message is decoded via the following mapping:
    // "1" -> 'A'
    // "2" -> 'B'
    // ...
    // "25" -> 'Y'
    // "26" -> 'Z'
    // However, while decoding the message, you realize that there are many different ways you can decode
    // the message because some codes are contained in other codes ("2" and "5" vs "25").
    // For example, "11106" can be decoded into:
    // "AAJF" with the grouping (1, 1, 10, 6)
    // "KJF" with the grouping (11, 10, 6)
    // The grouping (1, 11, 06) is invalid because "06" is not a valid code (only "6" is valid).
    // Note: there may be strings that are impossible to decode.
    // Given a string s containing only digits, return the number of ways to decode it. If the entire string cannot be decoded in any valid way, return 0.
    // DP + Memoization, recursive
    // O(n) time complexity
    #region 91. Decode Ways
    public static int NumDecodings(string s)
    {
        return NumDecodings(s, 0, new Dictionary<int, int>());
    }

    public static int NumDecodings(string s, int i, Dictionary<int, int> cache)
    {
        if (cache.ContainsKey(i)) return cache[i];
        var n = s.Length;
        if (i >= n) return 1;
        if (s[i] == '0') return 0;
        if (i == n - 1) return 1;
        if (s[i] == '1')
        {
            if (s[i + 1] == '0') // если 10, то мы не можем декодировать 1 и 0 по отдельности, поэтому идем сразу на i + 2
            {
                cache[i] = NumDecodings(s, i + 2, cache);
                return cache[i];
            }
            // Иначе, если 11 - 19, то мы можем декодировать 1 и следующий символ по отдельности, а также вместе как число от 10 до 19
            cache[i] = NumDecodings(s, i + 2, cache) + NumDecodings(s, i + 1, cache);
            return cache[i];
        }
        if (s[i] == '2')
        {
            // Если от 21 до 26, то мы можем декодировать 2 и следующий символ по отдельности, а также вместе как число от 20 до 26
            if (s[i + 1] >= '1' && s[i + 1] <= '6')
            {
                cache[i] = NumDecodings(s, i + 2, cache) + NumDecodings(s, i + 1, cache);
                return cache[i];
            }
            // Если 20, то мы не можем декодировать 2 и 0 по отдельности, поэтому идем сразу на i + 2
            if (s[i + 1] == '0')
            {
                cache[i] = NumDecodings(s, i + 2, cache);
                return cache[i];
            }
            // Иначе только один вариант декодирования - 2, нечего вычислять, идем дальше
        }

        var res = NumDecodings(s, i + 1, cache);
        cache[i] = res;
        return res;

        // обращение к cache[i] - это вычисление хеша, а cache у нас - это словарь, а не массив,
        // поэтому присваивание res избавляет нас от повторовного вычисления хеша.
        // на практике не имеет значения, но на LeetCode улучшает показатели времени выполнения
        // Иными словами, эквивалентный код имеет ту же самую асимптотику, но на LeetCode выдает 67%, против 100% с использованием временной переменной res:
        // cache[i] = NumDecodings(s, i + 1, cache);
        // return cache[i];
    }
    #endregion

    // #58
    // 347. Top K Frequent Elements
    // Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
    // Bucket sort
    // O(n) time complexity
    // O(n) space complexity
    #region 347. Top K Frequent Elements
    public static int[] TopKFrequent(int[] nums, int k)
    {
        // O(n)
        var freqInit = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++)
        {
            if (!freqInit.ContainsKey(nums[i]))
            {
                freqInit[nums[i]] = 1;
            }
            else
            {
                freqInit[nums[i]]++;
            }
        }

        // O(n)
        var freq = new Dictionary<int, List<int>>();
        foreach ((var key, var val) in freqInit)
        {
            if (!freq.ContainsKey(val))
            {
                freq[val] = new List<int>();
            }
            freq[val].Add(key);
        }

        // O(n), несмотря на вложенный цикл, так как суммарно мы пройдемся по всем элементам не более одного раза
        var result = new List<int>();
        for (int i = nums.Length; i > 0; i--)
        {
            if (freq.ContainsKey(i))
            {
                for (int j = 0; j < freq[i].Count && k > 0; j++)
                {
                    result.Add(freq[i][j]);
                    k--;
                }
            }
        }

        // O(k)
        return result.ToArray();
    }
    #endregion

    // #59
    // 253. Meeting Rooms II
    // Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.
    // Greedy + Min Heap
    // O(n log n) time complexity, O(n) space complexity
    // TODO: реализовать решение с сортировкой по началу и концам встреч, без использования кучи
    #region 253. Meeting Rooms II
    public static int MinMeetingRooms(int[][] intervals)
    {
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
        var result = 0;
        var rooms = new PriorityQueue<int, int>();
        for (int i = 0; i < intervals.Length; i++)
        {
            while (rooms.Count > 0 && rooms.Peek() <= intervals[i][0])
            {
                rooms.Dequeue();
            }

            rooms.Enqueue(intervals[i][1], intervals[i][1]);
            result = Math.Max(result, rooms.Count);
        }

        return result;
    }
    #endregion

    // #60
    // 98. Validate Binary Search Tree
    // Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    // A valid BST is defined as follows:
    // - The left subtree of a node contains only nodes with keys strictly less than the node's key.
    // - The right subtree of a node contains only nodes with keys strictly greater than the node's key.
    // - Both the left and right subtrees must also be binary search trees.
    #region 98. Validate Binary Search Tree
    public static bool IsValidBST(TreeNode root, long min = long.MinValue, long max = long.MaxValue) // long because root.val can be int.MinValue or int.MaxValue and its a valid value
    {
        if (root is null) return true;
        if (root.val <= min || root.val >= max) return false;

        return IsValidBST(root.left, min, root.val) && IsValidBST(root.right, root.val, max);
    }
    #endregion

    // #61
    // 226. Invert Binary Tree
    // Given the root of a binary tree, invert the tree, and return its root.
    #region 226. Invert Binary Tree
    public static TreeNode InvertTree(TreeNode root)
    {
        if (root is null) return null;

        var tmp = root.left;
        root.left = InvertTree(root.right);
        root.right = InvertTree(tmp);
        return root;
    }
    #endregion

    // #62
    // 100. Same Tree
    // Given the roots of two binary trees p and q, write a function to check if they are the same or not.
    // Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
    #region 100. Same Tree
    public static bool IsSameTree(TreeNode p, TreeNode q)
    {
        if (p is null && q is null) return true;
        if (p is null || q is null) return false;
        if (p.val != q.val) return false;
        return IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
    }
    #endregion

    // #63
    // 1143. Longest Common Subsequence
    // Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
    // A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted
    // without changing the relative order of the remaining characters.
    // For example, "ace" is a subsequence of "abcde".
    // A common subsequence of two strings is a subsequence that is common to both strings.
    //
    // 2D DP
    #region 1143. Longest Common Subsequence
    public static int LongestCommonSubsequence(string text1, string text2)
    {
        var n1 = text1.Length;
        var n2 = text2.Length;

        var dp = new int[n1 + 1, n2 + 1];

        for (int i = 1; i <= n1; i++)
        {
            for (int j = 1; j <= n2; j++)
            {
                dp[i, j] = text1[i - 1] == text2[j - 1]
                    ? dp[i - 1, j - 1] + 1
                    : Math.Max(dp[i - 1, j], dp[i, j - 1]);
            }
        }

        return dp[n1, n2];
    }
    #endregion

    // #64
    // 102. Binary Tree Level Order Traversal
    // Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
    // Обойти дерево по уровням
    // BFS
    // O(n) time complexity
    #region 102. Binary Tree Level Order Traversal
    public static IList<IList<int>> LevelOrder(TreeNode root)
    {
        var result = new List<IList<int>>();
        if (root is null) return result;

        var queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            var n = queue.Count;
            var levelResult = new List<int>();
            while (n > 0)
            {
                var node = queue.Dequeue();

                levelResult.Add(node.val);
                if (node.left is not null)
                {
                    queue.Enqueue(node.left);
                }
                if (node.right is not null)
                {
                    queue.Enqueue(node.right);
                }
                n--;
            }

            result.Add(levelResult);
        }
        return result;
    }
    #endregion

    // #65
    // 230. Kth Smallest Element in a BST
    // Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
    // O(h + k) time complexity, where h is the height of the tree, O(n) in the worst case
    // O(h) space complexity because of stack, O(n) in the worst case
    // Обход в глубину с помощью стека (итеративный in-order traversal): Левый - Корень - Правый
    #region 230. Kth Smallest Element in a BST
    public static int KthSmallest(TreeNode root, int k)
    {
        var stack = new Stack<TreeNode>();
        var counter = k;
        var current = root;
        while (current is not null || stack.Count > 0)
        {
            while (current is not null)
            {
                stack.Push(current);
                current = current.left;
            }

            current = stack.Pop();
            counter--;
            if (counter == 0) return current.val;
            current = current.right;
        }

        return -1; // this line should never be reached if k is valid, just to satisfy compiler
    }

    // Рекурсивный отбход с max heap - не оптимальный по памяти
    // public int KthSmallest(TreeNode root, int k, PriorityQueue<int, int> maxHeap = null)
    // {
    //     if (maxHeap is null)
    //     {
    //         maxHeap = new PriorityQueue<int, int>();
    //     }

    //     if (root is null) return -1;

    //     var leftResult = KthSmallest(root.left, k, maxHeap);
    //     if (leftResult >= 0) return leftResult;

    //     if (maxHeap.Count == k - 1)
    //     {
    //         return root.val;
    //     }

    //     maxHeap.Enqueue(root.val, -root.val);
    //     return KthSmallest(root.right, k, maxHeap);
    // }
    #endregion

    // #66
    // 104. Maximum Depth of Binary Tree
    // Given the root of a binary tree, return its maximum depth.
    // A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    #region 104. Maximum Depth of Binary Tree
    public static int MaxDepth(TreeNode root)
    {
        if (root is null) return 0;
        return 1 + Math.Max(MaxDepth(root.left), MaxDepth(root.right));
    }
    #endregion

    // #67
    // 105. Construct Binary Tree from Preorder and Inorder Traversal
    // Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree,
    // construct and return the binary tree.
    // То есть: даны два массива обхода дерева: прямой (корень - левый - правый) и симметричный (левый - корень - правый), без указания где листья.
    // Нужно восстановить дерево.
    // Идея: в прямом обходе первый элемент - это корень. Находим его в симметричном обходе, что дает нам размер левого и правого поддерева.
    // Зная размеры, рекурсивно восстанавливаем левое и правое поддерево, передавая соответствующие части массивов обходов.
    // Используем Span'ы чтобы избежать лишних аллокаций и удобно разрезать массив.
    // 
    // Далее неочевидный ход: запоминаем индексы всех элементов из inorder в словаре, чтобы избавиться от IndexOf на каждом шаге
    // + передаем offset, чтобы по исходному индексу понять какая у нас длина левого поддерева
    // Получается так: preorder всегда хранит корень, левое поддерево, правое поддерево. Длина левого поддерева - это индекс корня в inorder минус offset.
    // Тогда правое поддерево - это все, что после левого поддерева и корня в preorder.
    //
    // O(n) time complexity, так как каждый элемент дерева мы обрабатываем один раз
    #region 105. Construct Binary Tree from Preorder and Inorder Traversal
    public static TreeNode BuildTree(int[] preorder, int[] inorder)
    {
        var cache = new Dictionary<int, int>();
        for (int i = 0; i < inorder.Length; i++)
        {
            cache[inorder[i]] = i;
        }
        return BuildSubtree(preorder.AsSpan(), cache, 0);
    }

    private static TreeNode BuildSubtree(ReadOnlySpan<int> preorder, Dictionary<int, int> cache, int offset)
    {
        if (preorder.Length <= 0) return null;
        var root = preorder[0]; // Первый элемент всегда корень
        var leftSize = cache[root] - offset; // Размер левого поддерева
        var result = new TreeNode(root);
        result.left = BuildSubtree(preorder.Slice(1, leftSize), cache, offset); // Срез от 1 до leftSize - это левое поддерево
        result.right = BuildSubtree(preorder.Slice(leftSize + 1), cache, offset + leftSize + 1); // Slice до конца, т.к. все что осталось в preorder - это правое поддерево
        return result;
    }

    #region решение без словаря, с IndexOf, O(n^2) time complexity
    // Просто чтобы не забыть суть решения 
    public static TreeNode BuildTreeOn2(int[] preorder, int[] inorder)
    {
        return BuildSubtreeOn2(preorder.AsSpan(), inorder.AsSpan());
    }

    private static TreeNode BuildSubtreeOn2(Span<int> preorder, Span<int> inorder)
    {
        if (preorder.Length <= 0) return null;
        var root = preorder[0];
        var leftSize = inorder.IndexOf(root); // Вот от этого нам нужно было избавиться, так как он делает сложность O(n^2)
        var result = new TreeNode(root);
        result.left = BuildSubtreeOn2(preorder.Slice(1, leftSize), inorder.Slice(0, leftSize));
        result.right = BuildSubtreeOn2(preorder.Slice(leftSize + 1), inorder.Slice(leftSize + 1)); // Slice до конца, так что второй параметр не нужен
        return result;
    }
    #endregion

    #endregion

    // #68
    // 235. Lowest Common Ancestor of a Binary Search Tree
    // Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.
    // According to the definition of LCA on Wikipedia:
    // “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants
    // (where we allow a node to be a descendant of itself).”
    // BST, Binary Search Tree
    // O(h) time complexity, where h is the height of the tree
    #region 235. Lowest Common Ancestor of a Binary Search Tree

    // recursive
    public static TreeNode LowestCommonAncestor_BST(TreeNode root, TreeNode p, TreeNode q)
    {
        var l = p.val < q.val ? p.val : q.val;
        var r = p.val < q.val ? q.val : p.val;

        if (r < root.val) return LowestCommonAncestor_BST(root.left, p, q);
        if (l > root.val) return LowestCommonAncestor_BST(root.right, p, q);
        return root;
    }

    // iterative
    public static TreeNode LowestCommonAncestor_BST_Iterative(TreeNode root, TreeNode p, TreeNode q)
    {
        var l = p.val < q.val ? p.val : q.val;
        var r = p.val < q.val ? q.val : p.val;
        var curr = root;

        while (curr is not null)
        {
            if (r < curr.val)
            {
                curr = curr.left;
            }
            else if (l > curr.val)
            {
                curr = curr.right;
            }
            else
            {
                return curr;
            }
        }

        return root; // this line should never be reached if p and q are valid, just to satisfy compiler
    }
    #endregion

    // #69
    // 236. Lowest Common Ancestor of a Binary Tree
    // Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
    // According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T
    // that has both p and q as descendants (where we allow a node to be a descendant of itself).”
    // DFS, tree recursion
    // O(n) time complexity
    // Идея: возвращаем что нашли: либо p, либо q, либо LCA из каждого поддерева. Либо null, если ничего не нашли.
    #region 236. Lowest Common Ancestor of a Binary Tree
    public static TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
    {
        if (root is null || root == p || root == q) return root; // дошли до конца или нашли p или q
        var resL = LowestCommonAncestor(root.left, p, q); // Ищем в левом поддереве. Результат - либо p, либо q, либо LCA, либо null
        var resR = LowestCommonAncestor(root.right, p, q); // Ищем в правом поддереве. Результат - либо p, либо q, либо LCA, либо null
        if (resL is not null && resR is not null) return root; // в левом поддереве что-то нашли и в правом поддереве что-то нашли, значит p и q в разных поддеревьях, значит root общий предок
        return resL is null ? resR : resL; // результат либо в левом, либо в правом поддереве
    }
    #endregion

    // #70
    // 238. Product of Array Except Self
    // Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
    // The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
    // You must write an algorithm that runs in O(n) time and without using the division operation.
    // Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)
    // Каждый элемент результата равен произведению всех элементов, кроме себя самого
    // Prefix and Suffix products
    // O(n) time complexity
    // O(1) space complexity
    #region 238. Product of Array Except Self
    // Идея: использовать result как массив префиксных произведений со смещением на 1 (для удобства, чтоб не проверять if (i > 0)),
    // а суффиксное произведение накапливать в отдельной переменной
    public static int[] ProductExceptSelf(int[] nums)
    {
        var n = nums.Length;
        var result = new int[n];
        result[0] = 1;
        for (int i = 1; i < n; i++)
        {
            result[i] = result[i - 1] * nums[i - 1];
        }

        var suffix = 1;
        for (int i = n - 1; i >= 0; i--)
        {
            result[i] = result[i] * suffix;
            suffix *= nums[i];
        }

        return result;
    }

    // First attempt: using extra space for prefix and suffix arrays
    // NOT OPTIMAL FOR MEMORY
    // O(n) space complexity
    public static int[] ProductExceptSelfPrefixSuffix(int[] nums)
    {
        var n = nums.Length;
        var prefix = new int[n];
        var suffix = new int[n];
        var result = new int[n];
        prefix[0] = nums[0];
        suffix[n - 1] = nums[n - 1];
        for (int i = 1; i < n; i++)
        {
            prefix[i] = prefix[i - 1] * nums[i];
            suffix[n - 1 - i] = suffix[n - i] * nums[n - 1 - i];
        }

        for (int i = 0; i < n; i++)
        {
            var pref = i > 0 ? prefix[i - 1] : 1;
            var suff = i < n - 1 ? suffix[i + 1] : 1;
            result[i] = pref * suff;
        }

        return result;
    }

    #endregion

    // #71
    // 242. Valid Anagram
    // Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    #region 242. Valid Anagram
    public static bool IsAnagram(string s, string t)
    {
        if (s.Length != t.Length) return false;

        var freq = new Dictionary<char, int>();
        for (int i = 0; i < s.Length; i++)
        {
            if (freq.ContainsKey(s[i]))
            {
                freq[s[i]]++;
            }
            else
            {
                freq[s[i]] = 1;
            }
        }

        for (int i = 0; i < t.Length; i++)
        {
            if (!freq.ContainsKey(t[i]) || freq[t[i]] <= 0)
            {
                return false;
            }

            freq[t[i]]--;
        }

        return true;
    }
    #endregion

    // #72
    // 371. Sum of Two Integers
    // Given two integers a and b, return the sum of the two integers without using the operators + and -.
    // Bit manipulation, bitwise operations
    #region 371. Sum of Two Integers

    // My solution, first attempt
    // Идея: складываем побитово, учитывая перенос
    public static int GetSum(int a, int b)
    {
        var result = 0;
        var prevBit = 0;
        for (int i = 0; i < 32; i++)
        {
            var mask = (1 << i);
            var bitA = a & mask;
            var bitB = b & mask;
            var bitSum = (bitA ^ bitB) ^ prevBit; // XOR
            prevBit = ((bitA & bitB) > 0 || (bitA & prevBit) > 0 || (prevBit & bitB) > 0)
                ? mask << 1
                : 0;
            result |= bitSum;
        }
        return result;
    }

    // Vectorized solution using bitwise operations
    // Идея: складываем сразу всё без учета переноса с помощью XOR, а перенос вычисляем с помощью AND и сдвигаем влево
    // XOR дает сумму без учета переноса 1 ^ 1 = 0, 1 ^ 0 = 1, 0 ^ 0 = 0 - то есть как в обычном сложении, но теряется единица переноса
    // Дальше складываем результат с переносом, вычисляем новый перенос, и повторяем пока перенос не станет равен нулю, т.е. все перенесенные единицы учтены
    public static int GetSumVectorized(int a, int b)
    {
        while (b != 0)
        {
            var sum = a ^ b; // Складываем без учета переноса
            var carry = (a & b) << 1; // Вычисляем перенос и сдвигаем его влево, т.к. перенос идет на следующий разряд
            a = sum;
            b = carry;
        }
        return a;
    }
    #endregion

    // #73
    // 252. Meeting Rooms
    // Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.
    #region 252. Meeting Rooms
    public static bool CanAttendMeetings(int[][] intervals)
    {
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
        for (int i = 1; i < intervals.Length; i++)
        {
            if (intervals[i][0] < intervals[i - 1][1])
                return false;
        }

        return true;
    }
    #endregion

    // #74
    // 121. Best Time to Buy and Sell Stock
    // You are given an array prices where prices[i] is the price of a given stock on the ith day.
    // You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
    // Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
    #region 121. Best Time to Buy and Sell Stock
    public static int MaxProfit(int[] prices)
    {
        var result = 0;
        var min = int.MaxValue;
        for (int i = 0; i < prices.Length; i++)
        {
            if (prices[i] < min)
            {
                min = prices[i];
            }
            result = Math.Max(result, prices[i] - min);
        }
        return result;
    }
    #endregion

    // #75
    // 124. Binary Tree Maximum Path Sum
    // HARD
    // A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them.
    // A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
    // The path sum of a path is the sum of the node's values in the path.
    // Given the root of a binary tree, return the maximum path sum of any non-empty path.
    // DFS
    // O(n) time complexity
    //
    // Пояснение: путь не модет ветвиться, это должна быть одна линия
    // Идея: идеи рекурсивно. Возвращаем наверх максимум саб-пути через этот узел (root + Math.Max(left, right))
    // Отдельно высчитываем полный путь конкретно для этого узла (root + left + right), обновляем глобальный максимум
    // 
    // Потенциально опасное место: int.MinValue/4 в качестве -INFINITY. TODO: подумать, как это обойти.
    #region 124. Binary Tree Maximum Path Sum
    // First attempt
    public static int MaxPathSum(TreeNode root)
    {
        //if (root is null) return 0; // The number of nodes in the tree is in the range [1, 3 * 104].
        int _maxPathSum = int.MinValue / 4;
        MaxSubPathSum(root);
        return _maxPathSum;

        int MaxSubPathSum(TreeNode root)
        {
            if (root is null)
            {
                return int.MinValue / 4;
            }

            var leftSum = MaxSubPathSum(root.left);
            var rightSum = MaxSubPathSum(root.right);

            var maxSubSum = Math.Max(leftSum, rightSum);
            var result = Math.Max(root.val, root.val + maxSubSum);
            _maxPathSum = Math.Max(_maxPathSum, result);
            _maxPathSum = Math.Max(_maxPathSum, root.val + leftSum + rightSum);
            return result;
        }
    }
    #endregion

    // #76
    // 125. Valid Palindrome
    // A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters,
    // it reads the same forward and backward.
    // Alphanumeric characters include letters and numbers.
    // Given a string s, return true if it is a palindrome, or false otherwise.
    // Two Pointers
    #region 125. Valid Palindrome

    public static bool IsPalindrome(string s)
    {
        var l = 0;
        var r = s.Length - 1;
        while (l < r)
        {
            if (!IsValid125(s[l]))
            {
                l++;
                continue;
            }

            if (!IsValid125(s[r]))
            {
                r--;
                continue;
            }

            var cl = ToLower125(s[l]);
            var cr = ToLower125(s[r]);
            if (cl != cr) return false;
            l++;
            r--;
        }

        return true;
    }

    private static bool IsValid125(char c)
    {
        return (c >= '0' && c <= '9')
            || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z');
    }

    private static char ToLower125(char c)
    {
        return (c >= 'A' && c <= 'Z')
            ? (char)(c - 'A' + 'a')
            : c;
    }
    #endregion
}

public class Node133
{
    public int val;
    public IList<Node133> neighbors;

    public Node133()
    {
        val = 0;
        neighbors = new List<Node133>();
    }

    public Node133(int _val)
    {
        val = _val;
        neighbors = new List<Node133>();
    }

    public Node133(int _val, List<Node133> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }
}


public class TreeNode
{
    public int val;
    public TreeNode left;
    public TreeNode right;
    public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
    {
        this.val = val;
        this.left = left;
        this.right = right;
    }
 }