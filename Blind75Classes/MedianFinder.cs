namespace LeetcodePreapare;

public class MedianFinder
{
    private PriorityQueue<int, int> left;
    private PriorityQueue<int, int> right;

    public MedianFinder()
    {
        left = new PriorityQueue<int, int>();
        right = new PriorityQueue<int, int>();
    }

    public void AddNum(int num)
    {
        if (right.Count > 0 && num > right.Peek())
        {
            right.Enqueue(num, num);
        }
        else
        {
            left.Enqueue(num, -num);
        }

        if (left.Count > right.Count)
        {
            var lNum = left.Dequeue();
            right.Enqueue(lNum, lNum);
        }
        else if (right.Count > left.Count + 1)
        {
            var rNum = right.Dequeue();
            left.Enqueue(rNum, -rNum);
        }
    }

    public double FindMedian()
    {
        var leftNum = left.Count > 0 ? (double)left.Peek() : 0;
        var rightNum = right.Count > 0 ? (double)right.Peek() : 0;
        if (left.Count == right.Count)
        {
            return (leftNum + rightNum) / 2.0;
        }

        return rightNum;
    }
}
