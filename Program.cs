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

        // Maximum Subarray - LeetCode
        // Given an integer array nums, find the subarray with the largest sum, and return its sum.
        // Kadane’s Algorithm
        // Time complexity: O(n)
        public int MaxSubArray(int[] nums)
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

        // The Maximum Subarray - HackerRank
        // Given an array, find the maximum possible sum among: 1. all nonempty subarrays. 2.all nonempty subsequences.
        // We define subsequence as any subset of an array. We define a subarray as a contiguous subsequence in an array.
        // Kadane’s Algorithm
        // Time complexity: O(n)
        public static List<int> maxSubarray(List<int> arr)
        {
            var maxSubarraySum = arr[0];
            var currentSubarraySum = arr[0];
            var maxSunsequenceSum = arr[0];

            for (int i = 1; i < arr.Count(); i++)
            {
                currentSubarraySum = Math.Max(arr[i], currentSubarraySum + arr[i]);
                maxSubarraySum = Math.Max(currentSubarraySum, maxSubarraySum);

                maxSunsequenceSum = Math.Max(maxSunsequenceSum, maxSunsequenceSum + arr[i]);
                maxSunsequenceSum = Math.Max(maxSunsequenceSum, arr[i]);
            }

            return new List<int> { maxSubarraySum, maxSunsequenceSum };
        }

        // Sherlock and Anagrams - HackerRank
        #region sherlockAndAnagrams
        public static int sherlockAndAnagrams(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            var sunstrings = GetSubstringsFrequency(s);

            var count = 0;

            foreach (var item in sunstrings)
            {
                count += countPairs(item.Value);
            }

            return count;
        }

        private static int countPairs(int n)
        {
            // Pairs count: Ckn (C k from n) = n! / (n - k)! * k! 
            // C2n = n! / (n - 2)! * 2! = n * (n - 1) / 2
            return (n * (n - 1)) / 2;
        }

        private static Dictionary<string, int> GetSubstringsFrequency(string s)
        {
            var result = new Dictionary<string, int>();

            for (var i = 0; i < s.Length; i++)
            {
                var charCount = new int[26];

                for (var j = i; j < s.Length; j++)
                {
                    charCount[s[j] - 'a']++;
                    var key = string.Join('-', charCount);
                    if (result.ContainsKey(key))
                    {
                        result[key]++;
                    }
                    else
                    {
                        result[key] = 1;
                    }
                }
            }

            return result;
        }
        #endregion

        // Subarray Division - HackerRank
        public static int birthday(List<int> s, int d, int m)
        {
            if (s is null || s.Count() < m)
            {
                return 0;
            }

            var length = s.Count();
            var sumS = new int[length];
            var sum = 0;
            for (var i = 0; i < length; i++)
            {
                sum += s[i];
                sumS[i] = sum;
            }

            var count = 0;
            for (var i = 0; i < length - m + 1; i++)
            {
                var prevSum = i == 0 ? 0 : sumS[i - 1];
                if (sumS[i + m - 1] - prevSum == d)
                {
                    count++;
                }
            }

            return count;
        }

        // Sherlock and the Valid String - HackerRank
        // O(n) time complexity, O(1) space complexity
        public static string isValid(string s)
        {
            var charFrequenciesMap = new Dictionary<char, int>();
            for (var i = 0; i < s.Length; i++)
            {
                if (charFrequenciesMap.ContainsKey(s[i]))
                {
                    charFrequenciesMap[s[i]]++;
                }
                else
                {
                    charFrequenciesMap[s[i]] = 1;
                }
            }

            var freq = new Dictionary<int, int>();

            foreach (var item in charFrequenciesMap)
            {
                if (freq.ContainsKey(item.Value))
                {
                    freq[item.Value]++;
                }
                else
                {
                    freq[item.Value] = 1;
                }
            }

            if (freq.Keys.Count() == 1)
            {
                return "YES";
            }

            if (freq.Keys.Count() == 2)
            {
                var minKey = freq.Keys.First();
                var maxKey = freq.Keys.Last();
                if (minKey > maxKey)
                {
                    var tmp = minKey;
                    minKey = maxKey;
                    maxKey = tmp;
                }

                if (freq[maxKey] == 1 && maxKey - minKey == 1)
                {
                    return "YES";
                }

                if (freq[minKey] == 1 && minKey == 1)
                {
                    return "YES";
                }
            }

            return "NO";
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
            //int result = Result.birthday(new List<int>{ });


            Console.WriteLine(result);

            var arr = new[] { 1, 2 };
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
