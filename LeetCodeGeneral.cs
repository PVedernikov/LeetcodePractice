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
}
