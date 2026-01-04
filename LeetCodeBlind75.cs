using System.Collections.Generic;
using System.IO;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace LeetcodePreapare;

// https://leetcode.com/problem-list/r3q9lspc/
public static class LeetCodeBlind75
{
    // #1
    // 128. Longest Consecutive Sequence
    // Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
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


    // #2
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

    // #35

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

    // #38

    // #39

    // #40

    // #41
    // 322. Coin Change
    // You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
    // Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
    // You may assume that you have an infinite number of each kind of coin.
    // DP
    #region 322. Coin Change
    // 2D DP
    // NOT OPTIMAL FOR MEMORY
    // TODO: implement 1D DP solution, replace int.MaxValue with something to avoid unnecessary comparisons
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