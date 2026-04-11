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
}
