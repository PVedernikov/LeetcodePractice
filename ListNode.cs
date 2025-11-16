using System.Text;

namespace LeetcodePreapare;

//public class ListNode<T>
//{
//    public ListNode() { }

//    public ListNode(T value) { Value = value; }


//    public T Value { get; set; }

//    public ListNode<T> Next { get; set; }
//}

public class ListNode
{
    public ListNode() { }

    public ListNode(int value) { Value = value; }

    public int Value { get; set; }

    public ListNode next { get; set; }

    public ListNode Next
    {
        get
        {
            return next;
        }

        set
        {
            next = value;
        }
    }
}

public static class ListOperations
{
    public static ListNode Reverse(this ListNode head)
    {
        var currentNode = head;
        ListNode prevNode = null;
        while (currentNode is not null)
        {
            var tmpNext = currentNode.Next;
            currentNode.Next = prevNode;
            prevNode = currentNode;
            currentNode = tmpNext;
        }

        return prevNode;
    }

    public static ListNode ReverseWithDummyHead(this ListNode head)
    {
        var dummyHead = new ListNode();
        var currentNode = head;
        while (currentNode is not null)
        {
            var nextNode = dummyHead.Next;
            dummyHead.Next = currentNode;
            currentNode = currentNode.Next;
            dummyHead.Next.Next = nextNode;
        }

        return dummyHead.Next;
    }

    public static ListNode ReverseWithDummyHead2(this ListNode head)
    {
        var dummyHead = new ListNode();
        var currentNode = head;
        while (currentNode is not null)
        {
            var nextNode = currentNode.Next;
            dummyHead.InsertBehindHead(currentNode);
            currentNode = nextNode;
        }

        return dummyHead.Next;
    }

    public static ListNode InsertBehindHead(this ListNode head, int value)
    {
        var node = new ListNode(value);
        head.InsertBehindHead(node);
        return head;
    }

    public static ListNode InsertBehindHead(this ListNode head, ListNode node)
    {
        var nextNode = head.Next;
        head.Next = node;
        node.Next = nextNode;
        return head;
    }

    public static string GetString (this ListNode head)
    {
        if (head is null)
        {
            return string.Empty;
        }

        var current = head;
        var result = new StringBuilder();
        while (current is not null)
        {
            result.Append($"[{current.Value.ToString()}] -> ");
            current = current.Next;
        }

        return result.ToString();
    }
}