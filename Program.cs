namespace LeetcodePreapare
{
    class Result
    {
        // Pairs - HackerRank
        // Given an array of integers and a target value, determine the number of pairs of array elements that have a difference equal to the target value.
        // O(n log n) + O(n) time complexity due to sorting, O(1) space complexity.
        public static int pairs(int k, List<int> arr)
        {
            if (arr is null || arr.Count() < 2)
            {
                return 0;
            }

            arr.Sort();

            var left = 0;
            var right = 1;
            var pairsCount = 0;
            var length = arr.Count();

            while (left < length && right < length)
            {
                if (left >= right || arr[right] - arr[left] < k)
                {
                    right++;
                }

                if (right < length && arr[right] - arr[left] > k)
                {
                    left++;
                    right = left + 1;
                }

                if (left < length && right < length && arr[right] - arr[left] == k)
                {
                    pairsCount++;
                    right++;
                }
            }

            return pairsCount;
        }

        // Pairs - LeetCode
        // Given an integer array nums and an integer k, return the number of pairs (i, j) where i < j such that |nums[i] - nums[j]| == k.
        public static int CountKDifference(int k, int[] nums) //(int[] nums, int k)
        {
            if (nums is null || nums.Length < 2)
            {
                return 0;
            }

            Array.Sort(nums);

            var left = 0;
            var right = 1;
            var pairsCount = 0;
            var length = nums.Length;

            while (left < length)
            {
                if (left >= right || (right < length && nums[right] - nums[left] < k))
                {
                    right++;
                }

                if ((right < length && nums[right] - nums[left] > k) || right >= length)
                {
                    left++;
                    right = left + 1;
                }

                if (left < length && right < length && nums[right] - nums[left] == k)
                {
                    pairsCount++;
                    right++;
                }
            }

            return pairsCount;
        }


        // Minimum Absolute Difference in an Array - HackerRank
        // Given an array of integers, find the minimum absolute difference between any two elements in the array.
        public static int minimumAbsoluteDifference(List<int> arr)
        {
            arr.Sort();

            var left = 0;
            var minDiff = int.MaxValue;
            var length = arr.Count();

            while (left < length - 1)
            {
                var currDiff = arr[left + 1] - arr[left];

                if (currDiff < minDiff)
                {
                    minDiff = currDiff;
                }

                left++;
            }

            return minDiff;
        }


        // Merge two sorted linked lists - HackerRank
        // Given pointers to the heads of two sorted linked lists, merge them into a single, sorted linked list. Either head pointer may be null meaning that the corresponding list is empty.
        static SinglyLinkedListNode mergeLists(SinglyLinkedListNode head1, SinglyLinkedListNode head2)
        {
            if (head1 is null)
            {
                return head2;
            }

            if (head2 is null)
            {
                return head1;
            }

            var newHead = new SinglyLinkedListNode(int.MinValue);

            var currentNewNode = newHead;
            var currnetNode1 = head1;
            var currnetNode2 = head2;

            while (true)
            {
                if (currnetNode1 is null)
                {
                    currentNewNode.next = currnetNode2;
                    break;
                }

                if (currnetNode2 is null)
                {
                    currentNewNode.next = currnetNode1;
                    break;
                }

                if (currnetNode1.data <= currnetNode2.data)
                {
                    currentNewNode.next = currnetNode1;
                    currnetNode1 = currnetNode1.next;
                }
                else
                {
                    currentNewNode.next = currnetNode2;
                    currnetNode2 = currnetNode2.next;
                }
                currentNewNode = currentNewNode.next;
            }

            return newHead.next;
        }


    }

    internal class Program
    {
        static void Main(string[] args)
        {
            //string[] firstMultipleInput = Console.ReadLine().TrimEnd().Split(' ');

            //int n = Convert.ToInt32(firstMultipleInput[0]);

            //int k = Convert.ToInt32(firstMultipleInput[1]);

            //List<int> arr = Console.ReadLine().TrimEnd().Split(' ').ToList().Select(arrTemp => Convert.ToInt32(arrTemp)).ToList();

            //int result = Result.pairs(k, arr);
            int result = Result.CountKDifference(1, [1, 2, 2, 1]);

            Console.WriteLine(result);
        }
    }

    public class SinglyLinkedListNode
    {
        public SinglyLinkedListNode(int nodeData)
        {
            this.data = nodeData;
        }

        public int data { get; init; }
        public SinglyLinkedListNode next { get; set; }
    }
}
