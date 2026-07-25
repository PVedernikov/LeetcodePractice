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
        while (n > 0)
        {
            var a = 1;
            if (n <= 4)
            {
                a = n;
            }
            else
            {
                a = 3;
            }

            result *= a;
            n -= a;
        }

        return result;
    }
    #endregion
}
