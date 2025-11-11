using System.Xml.Linq;

namespace LeetcodePreapare;

// https://leetcode.com/problem-list/r3q9lspc/
public static class LeetCodeBlind75
{
    // #1
    // 1. Two Sum
    // HashSet
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

    // #2
    // 128. Longest Consecutive Sequence
    // HashSet
    // Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
    // O(n) time
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

    // #3
    // 128. Longest Consecutive Sequence
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