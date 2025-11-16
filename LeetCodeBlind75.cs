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

    // #6
    // 261. Graph Valid Tree
    // TODO: buy subscription

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
    // Linked List, Cycle Detection
    public static bool HasCycle(ListNode head)
    {
        var slow = head;
        var fast = head;

        while (fast is not null && fast.Next is not null)
        {
            slow = slow.Next;
            fast = fast.Next.Next;

            if (slow == fast) return true;
        }

        return false;
    }
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