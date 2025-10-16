namespace LeetcodePreapare;

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
    public static SinglyLinkedListNode mergeLists(SinglyLinkedListNode head1, SinglyLinkedListNode head2)
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


    // Two Characters - HackerRank
    // https://www.hackerrank.com/challenges/two-characters/problem
    #region Two Characters
    public static int alternate(string s)
    {
        var maxLength = 0;

        for (var i = 0; i < 26; i++)
        {
            for (var j = i + 1; j < 26; j++)
            {
                var length = GetLength(s, (char)(i + 'a'), (char)(j + 'a'));
                if (length > maxLength)
                {
                    maxLength = length;
                }
            }
        }

        return maxLength;
    }

    private static int GetLength(string s, char a, char b)
    {
        var expectedChar = '0';
        var length = 0;
        for (var i = 0; i < s.Length; i++)
        {
            if (s[i] == a || s[i] == b)
            {
                if (expectedChar != a && expectedChar != b)
                {
                    expectedChar = s[i] == a ? b : a;
                    length = 1;
                }
                else
                {
                    if (s[i] != expectedChar)
                    {
                        return 0;
                    }

                    expectedChar = s[i] == a ? b : a;

                    length++;
                }
            }
        }

        return length > 1 ? length : 0;
    }
    #endregion


    // Special String Again - HackerRank
    public static long substrCount(int n, string s)
    {
        if (string.IsNullOrEmpty(s) || n <= 0)
        {
            return 0;
        }

        long count = s.Length;
        var k = 0;
        while (k < s.Length)
        {
            long subCount = 0;
            var j = 1;
            while (k + j < s.Length && s[k] == s[k + j])
            {
                subCount++;
                j++;
            }

            // Substrings in a string with length = n formula:
            // n * (n + 1) / 2
            count += subCount * (subCount + 1) / 2;

            k += j;
        }

        for (var i = 1; i < s.Length - 1; i++)
        {
            long subCount = 0;
            var j = 1;
            while (i - j >= 0
                && i + j < s.Length
                && s[i - 1] == s[i + j]
                && s[i - 1] == s[i - j]
                && s[i - 1] != s[i])
            {
                subCount++;
                j++;
            }

            count += subCount;
        }

        return count;
    }


    // Common Child - HackerRank
    // Given two strings of equal length, what's the longest string that can be constructed such that it is a child of both?
    // Dynamic Programming, subsequence problem
    // O(n^2) time complexity, O(n^2) space complexity
    public static int commonChild(string s1, string s2)
    {
        var dp = new int[s1.Length + 1, s2.Length + 1];

        for (var i = 1; i <= s1.Length; i++)
        {
            for (var j = 1; j <= s2.Length; j++)
            {
                if (s1[i - 1] == s2[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                }
                else
                {
                    dp[i, j] = Math.Max(dp[i, j - 1], dp[i - 1, j]);
                }
            }
        }

        return dp[s1.Length, s2.Length];
    }

    // Making Anagrams - HackerRank
    #region Making Anagrams
    public static int makingAnagrams(string s1, string s2)
    {
        var frequenciesS1 = GetFrequencies(s1);
        var frequenciesS2 = GetFrequencies(s2);

        var count = 0;

        foreach (var frequencyS1 in frequenciesS1)
        {
            if (frequenciesS2.ContainsKey(frequencyS1.Key))
            {
                count += Math.Abs(frequenciesS2[frequencyS1.Key] - frequencyS1.Value);
            }
            else
            {
                count += frequencyS1.Value;
            }
        }

        foreach (var frequencyS2 in frequenciesS2)
        {
            if (!frequenciesS1.ContainsKey(frequencyS2.Key))
            {
                count += frequencyS2.Value;
            }
        }

        return count;
    }

    private static Dictionary<char, int> GetFrequencies(string s)
    {
        var frequencies = new Dictionary<char, int>();

        for (var i = 0; i < s.Length; i++)
        {
            if (frequencies.ContainsKey(s[i]))
            {
                frequencies[s[i]]++;
            }
            else
            {
                frequencies[s[i]] = 1;
            }
        }

        return frequencies;
    }
    #endregion


    // Bear and Steady Gene - HackerRank
    #region Bear and Steady Gene
    public static int steadyGene(string gene)
    {
        var quarter = gene.Length / 4;
        var freq = GetFrequenciesSteadyGene(gene);

        if (IsValidSubgene(freq, quarter))
        {
            return 0;
        }

        var l = 0;
        var result = int.MaxValue;
        for (var r = 0; r < gene.Length; r++)
        {
            freq[gene[r]]--;

            while (l <= r && IsValidSubgene(freq, quarter))
            {
                result = Math.Min(result, r - l + 1);
                freq[gene[l]]++;
                l++;
            }
        }

        return result;
    }

    private static bool IsValidSubgene(Dictionary<char, int> frequencies, int target)
    {
        foreach (var frequency in frequencies)
        {
            if (frequency.Value > target)
            {
                return false;
            }
        }

        return true;
    }

    private static Dictionary<char, int> GetFrequenciesSteadyGene(string s)
    {
        var result = new Dictionary<char, int>();
        for (int i = 0; i < s.Length; i++)
        {
            if (result.ContainsKey(s[i]))
            {
                result[s[i]]++;
            }
            else
            {
                result[s[i]] = 1;
            }
        }
        return result;
    }
    #endregion


    // Count Triplets - HackerRank
    public static long countTriplets(List<long> arr, long r)
    {
        var elementCount = new Dictionary<long, long>();
        var expectingCount = new Dictionary<long, long>();
        long arrLength = arr.Count();
        long result = 0;
        for (var i = 0; i < arrLength; i++)
        {
            if (expectingCount.ContainsKey(arr[i]))
            {
                result += expectingCount[arr[i]];
            }

            var leftElement = arr[i] / r;
            if (arr[i] % r == 0 && elementCount.ContainsKey(leftElement))
            {
                if (expectingCount.ContainsKey(arr[i] * r))
                {
                    expectingCount[arr[i] * r] += elementCount[leftElement];
                }
                else
                {
                    expectingCount[arr[i] * r] = elementCount[leftElement];
                }
            }

            if (elementCount.ContainsKey(arr[i]))
            {
                elementCount[arr[i]]++;
            }
            else
            {
                elementCount[arr[i]] = 1;
            }
        }
        return result;
    }

    // Frequency Queries - HackerRank
    public static List<int> freqQuery(List<List<int>> queries)
    {
        var result = new List<int>();

        var frequencies = new Dictionary<int, int>();
        var elements = new Dictionary<int, int>();
        //var elements = new List<int>();

        foreach (var query in queries)
        {
            if (query[0] == 1)
            {
                if (elements.ContainsKey(query[1]))
                {
                    if (frequencies.ContainsKey(elements[query[1]]) && frequencies[elements[query[1]]] > 0)
                    {
                        frequencies[elements[query[1]]]--;
                    }

                    elements[query[1]]++;

                    if (frequencies.ContainsKey(elements[query[1]]))
                    {
                        frequencies[elements[query[1]]]++;
                    }
                    else
                    {
                        frequencies[elements[query[1]]] = 1;
                    }
                }
                else
                {
                    elements[query[1]] = 1;

                    if (frequencies.ContainsKey(1))
                    {
                        frequencies[1]++;
                    }
                    else
                    {
                        frequencies[1] = 1;
                    }
                }
            }

            if (query[0] == 2)
            {
                if (elements.ContainsKey(query[1]) && elements[query[1]] > 0)
                {
                    if (frequencies.ContainsKey(elements[query[1]]) && frequencies[elements[query[1]]] > 0)
                    {
                        frequencies[elements[query[1]]]--;
                    }

                    elements[query[1]]--;

                    if (frequencies.ContainsKey(elements[query[1]]))
                    {
                        frequencies[elements[query[1]]]++;
                    }
                    else
                    {
                        frequencies[elements[query[1]]] = 1;
                    }
                }
            }

            if (query[0] == 3)
            {
                if (frequencies.ContainsKey(query[1]) && frequencies[query[1]] > 0)
                {
                    result.Add(1);
                }
                else
                {
                    result.Add(0);
                }
            }
        }

        return result;
    }

    // Closest Numbers - HackerRank
    public static List<int> closestNumbers(List<int> arr)
    {
        arr.Sort();

        var length = arr.Count;
        var absoluteMin = int.MaxValue;
        var result = new List<int>();

        for (var i = 0; i < arr.Count - 1; i++)
        {
            var currentMin = arr[i + 1] - arr[i];
            if (currentMin == absoluteMin)
            {
                result.Add(arr[i]);
                result.Add(arr[i + 1]);
            }

            if (currentMin < absoluteMin)
            {
                absoluteMin = currentMin;
                result.Clear();
                result.Add(arr[i]);
                result.Add(arr[i + 1]);
            }
        }

        return result;
    }


    // Missing Numbers - HackerRank
    #region Missing Numbers
    public static List<int> missingNumbers(List<int> arr, List<int> brr)
    {
        var freqA = GetFrequenciesMissingNumbers(arr);
        var freqB = GetFrequenciesMissingNumbers(brr);

        var missing = new List<int>();

        foreach (var itemB in freqB)
        {
            if (freqA.ContainsKey(itemB.Key))
            {
                if (freqA[itemB.Key] < itemB.Value)
                {
                    missing.Add(itemB.Key);
                }
            }
            else
            {
                missing.Add(itemB.Key);
            }
        }
        missing.Sort();

        return missing;
    }

    private static Dictionary<int, int> GetFrequenciesMissingNumbers(List<int> arr)
    {
        var result = new Dictionary<int, int>();

        for (int i = 0; i < arr.Count; i++)
        {
            if (result.ContainsKey(arr[i]))
            {
                result[arr[i]]++;
            }
            else
            {
                result[arr[i]] = 1;
            }
        }

        return result;
    }
    #endregion

    // Binary Search Tree : Lowest Common Ancestor - HackerRank
    #region Binary Search Tree : Lowest Common Ancestor
    public static TreeNodeLCA LCA_Recursive(TreeNodeLCA root, int a, int b)
    {
        if (root is null)
        {
            return null;
        }

        if (a > b)
        {
            var tmp = a;
            a = b;
            b = tmp;
        }

        if (a <= root.Value && root.Value <= b)
        {
            return root;
        }

        if (a < root.Value && b < root.Value)
        {
            return LCA_Recursive(root.Left, a, b);
        }

        return LCA_Recursive(root.Right, a, b);
    }

    public static TreeNodeLCA LCA(TreeNodeLCA root, int a, int b)
    {
        if (a > b)
        {
            var tmp = a;
            a = b;
            b = tmp;
        }

        var ancestor = root;
        while (ancestor != null)
        {
            if (a <= ancestor.Value && ancestor.Value <= b)
            {
                return ancestor;
            }

            if (b < root.Value)
            {
                ancestor = ancestor.Left;
            }
            else
            {
                ancestor = ancestor.Right;
            }
        }

        return ancestor;
    }
    #endregion

    // Max Min (Angry Children) - HackerRank
    // Sliding window problem
    public static int maxMin(int k, List<int> arr)
    {
        arr.Sort();

        var min = arr[0];
        var max = arr[k - 1];

        for (var i = 1; i <= arr.Count - k; i++)
        {
            if (max - min > arr[i + k - 1] - arr[i])
            {
                min = arr[i];
                max = arr[i + k - 1];
            }
        }

        return max - min;
    }

    // Lily's Homework - HackerRank
    // Index Mapping
    public static int lilysHomework(List<int> arr)
    {
        var indexMap1 = new Dictionary<int, int>();
        var indexMap2 = new Dictionary<int, int>();
        var copyArr1 = new List<int>();
        var copyArr2 = new List<int>();

        for (int i = 0; i < arr.Count; i++)
        {
            indexMap1[arr[i]] = i;
            indexMap2[arr[i]] = i;
            copyArr1.Add(arr[i]);
            copyArr2.Add(arr[i]);
        }

        var ascArr = copyArr1.OrderBy(i => i).ToArray();
        var resultAsc = 0;
        for (int i = 0; i < copyArr1.Count; i++)
        {
            if (copyArr1[i] == ascArr[i])
            {
                continue;
            }

            var correctI = indexMap1[ascArr[i]];
            var tmp = copyArr1[i];
            copyArr1[i] = copyArr1[correctI];
            copyArr1[correctI] = tmp;

            indexMap1[copyArr1[correctI]] = correctI;
            indexMap1[copyArr1[i]] = i;

            resultAsc++;
        }

        var descArr = copyArr2.OrderByDescending(i => i).ToArray();
        var resultDesc = 0;
        for (int i = 0; i < copyArr2.Count; i++)
        {
            if (copyArr2[i] == descArr[i])
            {
                continue;
            }

            var correctI = indexMap2[descArr[i]];
            var tmp = copyArr2[i];
            copyArr2[i] = copyArr2[correctI];
            copyArr2[correctI] = tmp;

            indexMap2[copyArr2[correctI]] = correctI;
            indexMap2[copyArr2[i]] = i;

            resultDesc++;
        }

        return Math.Min(resultAsc, resultDesc);
    }

    // Balanced Brackets - HackerRank
    public static string isBalanced(string s)
    {
        var openingBrackets = new Stack<char>();

        for (var i = 0; i < s.Length; i++)
        {
            if (s[i] == '(' || s[i] == '{' || s[i] == '[')
            {
                openingBrackets.Push(s[i]);
            }

            if (s[i] == ')' || s[i] == '}' || s[i] == ']')
            {
                if (openingBrackets.Count <= 0)
                {
                    return "NO";
                }

                var openingBracket = openingBrackets.Pop();

                if ((s[i] == ')' && openingBracket != '(')
                    || (s[i] == '}' && openingBracket != '{')
                    || (s[i] == ']' && openingBracket != '['))
                {
                    return "NO";
                }
            }
        }

        if (openingBrackets.Count > 0)
        {
            return "NO";
        }

        return "YES";
    }

    // Reverse a linked list - HackerRank
    public static SinglyLinkedListNode reverse(SinglyLinkedListNode llist)
    {
        var tmpHead = new SinglyLinkedListNode(0);

        while (llist is not null)
        {
            // insert element behind the head
            var tmp = tmpHead.next;
            tmpHead.next = llist;
            llist = llist.next;
            tmpHead.next.next = tmp;
        }

        return tmpHead.next;
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
        //int result = Result.CountKDifference(1, [1, 2, 2, 1]);
        //int result = Result.birthday(new List<int>{ });

        var arr = new List<int> { 100, 200, 300, 350, 400, 401, 402 };

        int result = Result.maxMin(3, arr);

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

public class TreeNodeLCA
{
    public int Value { get; set; }
    public TreeNodeLCA Left { get; set; }
    public TreeNodeLCA Right { get; set; }

    public TreeNodeLCA(int value)
    {
        Value = value;
    }

    public void Insert(int value)
    {
        if (value < Value)
        {
            if (Left is null)
            {
                Left = new TreeNodeLCA(value);
            }
            else
            {
                Left.Insert(value);
            }
        }
        else
        {
            if (Right is null)
            {
                Right = new TreeNodeLCA(value);
            }
            else
            {
                Right.Insert(value);
            }
        }
    }
}


public class TextEditorOperation
{
    public int Operation { get; set; }
    public int Number { get; set; }
    public char[] Chars { get; set; } = Array.Empty<char>();
}

public class TextEditor
{
    private Stack<TextEditorOperation> undoOperations = new Stack<TextEditorOperation>();
    private char[] chars = new char[1000000];
    private int currentLength = 0;

    public char Do(string operation)
    {
        var op = GetOperationData(operation);

        return Do(op, false);
    }

    private char Do(TextEditorOperation action, bool isUndoOperation = false)
    {
        if (action.Operation == 1)
        {
            if (!isUndoOperation)
            {
                undoOperations.Push(new TextEditorOperation
                {
                    Operation = 2,
                    Number = action.Chars.Length
                });
            }
            foreach (var symbol in action.Chars)
            {
                if (currentLength < chars.Length)
                {
                    chars[currentLength] = symbol;
                }

                currentLength++;
            }
        }

        if (action.Operation == 2)
        {
            var actualLength = Math.Min(action.Number, currentLength);
            if (!isUndoOperation)
            {
                var addChars = new char[actualLength];
                for (int i = 0; i < actualLength; i++)
                {
                    addChars[i] = chars[currentLength - actualLength + i];
                }

                undoOperations.Push(new TextEditorOperation
                {
                    Operation = 1,
                    Number = 0,
                    Chars = addChars
                });
            }

            currentLength -= actualLength;
        }

        if (action.Operation == 3)
        {
            if (action.Number <= currentLength && action.Number > 0)
            {
                return chars[action.Number - 1];
            }
        }

        if (action.Operation == 4)
        {
            var op = undoOperations.Pop();
            Do(op, true);
        }

        return '0';
    }

    private TextEditorOperation GetOperationData(string input)
    {
        var operationData = input.Trim().Split(' ');
        if (int.TryParse(operationData[0], out int operationCode))
        {
            if (operationCode == 1) // Add symbols
            {
                var chars = operationData[1].ToCharArray();
                return new TextEditorOperation
                {
                    Operation = 1,
                    Number = 0,
                    Chars = chars
                };
            }

            if (operationCode == 2 || operationCode == 3) // Delete or print
            {
                if (int.TryParse(operationData[1], out int value))
                {
                    return new TextEditorOperation
                    {
                        Operation = operationCode,
                        Number = value,
                        Chars = Array.Empty<char>()
                    };
                }
            }

            return new TextEditorOperation // Undo
            {
                Operation = 4,
                Number = 0,
                Chars = Array.Empty<char>()
            };
        }

        return new TextEditorOperation
        {
            Operation = 0,
            Number = 0,
            Chars = Array.Empty<char>()
        };
    }
}