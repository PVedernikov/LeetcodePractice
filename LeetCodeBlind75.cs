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
    // TODO: переписать под память O(1). Идея: найти центр списка (fast & slow указатели), развернуть вторую половину, затем слить две половины
    public static void ReorderList(ListNode head)
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

    // #14
    // 269. Alien Dictionary
    // TODO: buy subscription

    // #15
    // 271. Encode and Decode Strings
    // TODO: buy subscription

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


    // #?
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


    // #?
    // 242. Valid Anagram
    // Given two strings s and t, return true if t is an anagram of s, and false otherwise.
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