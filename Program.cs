using System;
using System.Net;
using System.Reflection;

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
        var maxArrSum = arr[0];
        var maxSeqSum = arr[0];
        var currSum = arr[0];
        for (var i = 1; i < arr.Count; i++)
        {
            currSum = Math.Max(arr[i], currSum + arr[i]);
            maxArrSum = Math.Max(currSum, maxArrSum);

            maxSeqSum = Math.Max(maxSeqSum, maxSeqSum + arr[i]);
            maxSeqSum = Math.Max(maxSeqSum, arr[i]);
        }

        return new List<int> { maxArrSum, maxSeqSum };
    }

    // Sherlock and Anagrams - HackerRank
    // https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem
    // To solve the HackerRank problem "Sherlock and Anagrams,"
    // you need to find the number of pairs of substrings within a given string that are anagrams of each other.
    // The most efficient way to do this is using a hash map (or dictionary) to store a canonical representation of each substring.
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
    // Sherlock considers a string to be valid if all characters of the string appear the same number of times.
    // It is also valid if he can remove just 1 character at 1 index in the string, and the remaining characters will occur the same number of times.
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
    // Given two strings, s1 and s2, that may not be of the same length,
    // determine the minimum number of character deletions required to make s1 and s2 anagrams.
    // Any characters can be deleted from either of the strings.
    // HashMap
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
    // Given a string GENE, can you help Limak find the length of the smallest possible substring that he can replace to make GENE a steady gene?
    // Where all the characters occur exactly n/4 times.
    // Sliding window problem
    // Идея: Двигаем правую границу пока не станет валидно, затем двигаем левую границу пока не станет невалидно
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
    // You are given an array and you need to find number of tripets of indices
    // such that the elements at those indices are in geometric progression for a given common ratio
    // https://www.hackerrank.com/challenges/count-triplets-1/problem
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
    // Given two arrays of integers, find which elements in the second array are missing from the first array.
    // HashMap
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
    // Составить массив размера k из массива arr, такой что разница между максимальным и минимальным элементом была минимальна
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
    // Сколько перестановок нужно сделать, чтобы массив стал упорядочен по возрастанию или убыванию
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

    // The Coin Change Problem - HackerRank
    // 2DP, составить сумму из монет
    #region The Coin Change Problem
    public static long getWays(int n, List<long> c)
    {
        var coins = new List<long> { 0 };
        coins.AddRange(c);

        var dp = new long[coins.Count, n + 1];

        for (var i = 0; i < coins.Count; i++)
        {
            dp[i, 0] = 1;
        }
        for (var j = 1; j <= n; j++)
        {
            dp[0, j] = 0;
        }

        for (var i = 1; i < coins.Count; i++)
        {
            for (var j = 1; j <= n; j++)
            {
                if (coins[i] > j)
                {
                    dp[i, j] = dp[i - 1, j];
                }
                else
                {
                    dp[i, j] = dp[i - 1, j] + dp[i, j - coins[i]];
                }
            }
        }

        return dp[coins.Count - 1, n];
    }

    //public static long getWays(int n, List<long> c)
    //{
    //    var dp = new long[n + 1];
    //    dp[0] = 1;
    //    foreach (var coin in c)
    //    {
    //        for (var amount = (int)coin; amount <= n; amount++)
    //        {
    //            dp[amount] += dp[amount - (int)coin];
    //        }
    //    }
    //    return dp[n];
    //}
    #endregion

    // Unbounded Knapsack - HackerRank
    // DP
    // Given an array of integers and a target sum, determine the sum nearest to but not exceeding the target that can be created.
    // To create the sum, use any element of your array zero or more times.
    public static int unboundedKnapsack(int k, List<int> arr)
    {
        var dp = new int[k + 1];
        dp[0] = 0;

        for (int i = 0; i < arr.Count; i++)
        {
            for (var j = arr[i]; j <= k; j++)
            {
                if (j >= arr[i])
                {
                    dp[j] = Math.Max(dp[j], arr[i] + dp[j - arr[i]]);
                }

            }
        }

        return dp[k];
    }

    // Candies - HackerRank
    // Alice wants to give at least 1 candy to each child.
    // If two children sit next to each other, then the one with the higher rating must get more candies. Alice wants to minimize the total number of candies she must buy.
    public static long candies(int n, List<int> arr)
    {
        var candies = new long[arr.Count];

        for (int i = 0; i < arr.Count; i++)
        {
            candies[i] = 1;
        }

        for (int i = 1; i < arr.Count; i++)
        {
            if (arr[i] > arr[i - 1])
            {
                candies[i] = candies[i - 1] + 1;
            }
        }

        for (int i = arr.Count - 2; i >= 0; i--)
        {
            if (arr[i] > arr[i + 1])
            {
                candies[i] = Math.Max(candies[i], candies[i + 1] + 1);
            }
        }


        return candies.Sum();
    }

    // Abbreviation - HackerRank
    // https://www.hackerrank.com/challenges/abbr/problem
    // Вам даются две строки A и B. Вам нужно преобразовать строку A в два шага:
    // 1. Сделать некоторые маленькие буквы большими,
    // 2. Удалить оставшиеся маленькие буквы, так, чтобы получить в итоге строку B.
    // Для каждого тестового случая выведите "YES", если возможно преобразовать строку A в строку B и "NO", если невозможно.
    // A = daBcd, B = ABC
    // Result: YES
    // ВАЖНО: Заглавные символы удалять нельзя, по ним должно быть точное соответствие
    public static string abbreviation(string a, string b)
    {
        var dp = new bool[a.Length + 1, b.Length + 1];
        var can = true;
        dp[0, 0] = true;
        for (var i = 1; i < a.Length + 1; i++)
        {
            if (a[i - 1] >= 'A' && a[i - 1] <= 'Z')
            {
                can = false;
            }

            dp[i, 0] = can;
        }

        for (var j = 1; j < b.Length + 1; j++)
        {
            dp[0, j] = false;
        }

        for (var i = 1; i < a.Length + 1; i++)
        {
            for (var j = 1; j < b.Length + 1; j++)
            {
                if (a[i - 1] >= 'A' && a[i - 1] <= 'Z')
                {

                    if (a[i - 1] == b[j - 1])
                    {
                        dp[i, j] = dp[i - 1, j - 1];
                    }
                    else
                    {
                        dp[i, j] = false;
                    }
                }
                else
                {
                    var bigA = (char)(a[i - 1] - ('a' - 'A'));
                    if (a[i - 1] == b[j - 1] || bigA == b[j - 1])
                    {
                        dp[i, j] = dp[i - 1, j - 1] || dp[i - 1, j]; ;
                    }
                    else
                    {
                        dp[i, j] = dp[i - 1, j];
                    }
                }
            }
        }

        return dp[a.Length, b.Length] ? "YES" : "NO";
    }

    // Journey to the Moon - HackerRank
    // Determine how many pairs of astronauts from different countries they can choose from.
    // DFS, Graph, список смежности 
    // Рекурсия через стек
    #region Journey to the Moon
    public static long journeyToMoon(int n, List<List<int>> astronaut)
    {
        var connections = new List<int>[n];
        var visited = new bool[n];

        for (int i = 0; i < n; i++)
        {
            connections[i] = new List<int>();
        }

        foreach (var pair in astronaut)
        {
            connections[pair[0]].Add(pair[1]);
            connections[pair[1]].Add(pair[0]);
        }

        var groups = new List<long>();

        for (int i = 0; i < n; i++)
        {
            var size = CountGroupSize(i, visited, connections);
            if (size > 1) // Если в группе 1 человек, нет смысла считать. Если 0 - тем более
            {
                groups.Add(size);
            }
        }
        long lN = (long)n;
        // C2n = n * (n - 1) / 2
        long totalPairs = (lN * (lN - 1)) / 2;
        long sameCountryPairs = 0;
        foreach (var g in groups)
        {
            sameCountryPairs += (g * (g - 1)) / 2;
        }
        long result = totalPairs - sameCountryPairs;
        return result;
    }

    private static long CountGroupSize(int id, bool[] visited, List<int>[] connections)
    {
        long result = 0;
        var stack = new Stack<int>();
        stack.Push(id);

        while (stack.Count > 0)
        {
            var cId = stack.Pop();
            if (visited[cId])
            {
                continue;
            }

            result++;
            visited[cId] = true;

            foreach (var ccId in connections[cId])
            {
                stack.Push(ccId);
            }
        }

        return result;
    }
    
    //public static long journeyToMoon(int n, List<List<int>> astronaut)
    //{
    //    var connections = new List<int>[n];
    //    var visited = new bool[n];

    //    foreach (var pair in astronaut)
    //    {
    //        if (connections[pair[0]] is null)
    //        {
    //            connections[pair[0]] = new List<int>();
    //        }
    //        if (connections[pair[1]] is null)
    //        {
    //            connections[pair[1]] = new List<int>();
    //        }

    //        connections[pair[0]].Add(pair[1]);
    //        connections[pair[1]].Add(pair[0]);
    //    }

    //    long result = 0;
    //    long sum = 0;
    //    for (var i = 0; i < n; i++)
    //    {
    //        if (visited[i])
    //        {
    //            continue;
    //        }

    //        long groupSize = CountNeigbours(i, visited, connections);
    //        result += groupSize * sum;
    //        sum += groupSize;
    //    }

    //    return result;
    //}

    //private static long CountNeigbours(int id, bool[] visited, List<int>[] connections)
    //{
    //    if (visited[id])
    //        return 0;

    //    visited[id] = true;

    //    long result = 1;
    //    if (connections[id] is null || connections[id].Count() == 0)
    //    {
    //        return result;
    //    }

    //    foreach (var nid in connections[id])
    //    {
    //        result += CountNeigbours(nid, visited, connections);
    //    }

    //    return result;
    //}

    //private static long CountNeighboursIterative(int start, bool[] visited, List<int>[] connections)
    //{
    //    long count = 0;
    //    var stack = new Stack<int>();
    //    stack.Push(start);

    //    while (stack.Count > 0)
    //    {
    //        var node = stack.Pop();

    //        if (visited[node])
    //            continue;

    //        visited[node] = true;
    //        count++;

    //        if (connections[node] != null)
    //        {
    //            foreach (var neighbor in connections[node])
    //            {
    //                if (!visited[neighbor])
    //                {
    //                    stack.Push(neighbor);
    //                }
    //            }
    //        }
    //    }

    //    return count;
    //}
    #endregion


    // Roads and Libraries - HackerRank
    // Идея: если библиотека дешевле дороги, просто строим библиотеки в каждом городе
    // Иначе считаем связные компоненты графа и для каждой строим 1 библиотеку и дороги ко всем остальным городам в компоненте
    // DFS, Graph, список смежности
    // Рекурсивный и итеративный варианты обхода графа 
    #region Roads and Libraries
    public static long roadsAndLibraries(int n, int c_lib, int c_road, List<List<int>> cities)
    {
        if (c_lib < c_road)
        {
            return (long)c_lib * (long)n;
        }

        var linkedCities = new List<int>[n];
        var visited = new bool[n];

        // init list
        for (var i = 0; i < n; i++)
        {
            linkedCities[i] = new List<int>();
        }

        foreach (var pair in cities)
        {
            // Important! cities are 1-based indexed
            // so actual index is pair[i] - 1
            linkedCities[pair[0] - 1].Add(pair[1] - 1);
            linkedCities[pair[1] - 1].Add(pair[0] - 1);
        }

        // counting groups and their sizes
        var groupSizes = new List<long>();
        for (var i = 0; i < n; i++)
        {
            var size = CountLinksIterative(i, visited, linkedCities);
            if (size > 0)
            {
                groupSizes.Add(size);
            }
        }

        // If library is more expensive, build 1 library and connect other cities with roads
        long result = 0;
        foreach (var size in groupSizes)
        {
            result += c_lib + c_road * (size - 1);
        }

        return result;
    }

    private static long CountLinksRecursive(int id, bool[] visited, List<int>[] linkedCities)
    {
        if (visited[id])
        {
            return 0;
        }

        long result = 1;
        visited[id] = true;

        if (linkedCities[id] is not null && linkedCities[id].Count > 0)
        {
            foreach (var linkedCityId in linkedCities[id])
            {
                result += CountLinksRecursive(linkedCityId, visited, linkedCities);
            }
        }

        return result;
    }

    private static long CountLinksIterative(int id, bool[] visited, List<int>[] linkedCities)
    {
        var cities = new Stack<int>();
        cities.Push(id);

        long result = 0;
        while (cities.Count > 0)
        {
            var cityId = cities.Pop();
            if (visited[cityId])
            {
                continue;
            }

            visited[cityId] = true;
            result++;

            if (linkedCities[cityId] is not null && linkedCities[cityId].Count > 0)
            {
                foreach (var linkedCityId in linkedCities[cityId])
                {
                    cities.Push(linkedCityId);
                }
            }
        }

        return result;
    }
    #endregion

    // Even Tree - HackerRank
    // DFS, Graph, список смежности
    // ВАЖНО: DFS реализован не идеально, подумать как улучшить
    #region Even Tree
    public static int evenForest(int t_nodes, int t_edges, List<int> t_from, List<int> t_to)
    {
        var tree = new List<int>[t_nodes];
        var countedSubtrees = new int[t_nodes];

        for (var i = 0; i < tree.Length; i++)
        {
            tree[i] = new List<int>();
            countedSubtrees[i] = -1;
        }

        for (var i = 0; i < t_edges; i++)
        {
            tree[t_to[i] - 1].Add(t_from[i] - 1);
        }

        var result = 0;
        for (var i = 1; i < tree.Length; i++) // can't cut root, so start from 1
        {
            var nodesCount = GetNodesCount(i, tree, countedSubtrees);
            if (nodesCount % 2 == 0)
            {
                result++;
            }
        }
        return result;
    }

    private static int GetNodesCount(int id, List<int>[] tree, int[] countedSubtrees)
    {
        if (countedSubtrees[id] >= 0)
        {
            return countedSubtrees[id];
        }

        var nodes = new Stack<int>();
        nodes.Push(id);

        var result = 0;

        while (nodes.Count > 0)
        {
            var rootId = nodes.Pop();

            result++; // count root element

            if (tree[rootId].Count > 0)
            {
                foreach (var childId in tree[rootId])
                {
                    nodes.Push(childId); // add children to count
                }
            }
        }
        countedSubtrees[id] = result;
        return result;
    }
    #endregion


    // Jesse and Cookies - HackerRank
    // Min-Heap (Priority Queue)
    // https://www.hackerrank.com/challenges/jesse-and-cookies/problem
    public static int cookies(int k, List<int> A)
    {
        var sweets = new PriorityQueue<int, int>();

        foreach (var sweet in A)
        {
            sweets.Enqueue(sweet, sweet);
        }

        var result = 0;


        while (sweets.Peek() < k && sweets.Count > 1)
        {
            var lowestSweet = sweets.Dequeue();
            var secondLowestSweet = sweets.Dequeue();
            var newSweet = lowestSweet + 2 * secondLowestSweet;
            sweets.Enqueue(newSweet, newSweet);

            result++;
        }

        return sweets.Peek() < k ? -1 : result;
    }

    // Luck Balance - HackerRank
    // Greedy Algorithm
    // DESC sort
    public static int luckBalance(int k, List<List<int>> contests)
    {
        var importantContests = new List<int>();
        var luck = 0;

        foreach (var contest in contests)
        {
            if (contest[1] == 0)
            {
                luck += contest[0];
            }
            else
            {
                importantContests.Add(contest[0]);
            }
        }

        importantContests.Sort((a, b) => b.CompareTo(a));


        for (var i = 0; i < importantContests.Count; i++)
        {
            if (i < k)
            {
                luck += importantContests[i];
            }
            else
            {
                luck -= importantContests[i];
            }
        }

        return luck;
    }


    // Minimum Average Waiting Time - HackerRank
    // Min-Heap (Priority Queue), Greedy Algorithm
    // Идея: сортируем клиентов по времени, затем двигаем время когда очередная пицца готова,
    // затем добавляем в очередь всех клиентов, которые пришли к этому времени
    // выбираем из PriorityQueue следующий заказ с минимальным временем приготовления
    public static long minimumAverage(List<List<long>> customers)
    {
        customers.Sort((a, b) => a[0].CompareTo(b[0]));

        var customersQueue = new Queue<List<long>>();
        for (int i = 0; i < customers.Count; i++)
        {
            customersQueue.Enqueue(customers[i]);
        }
        var ordersTaken = new PriorityQueue<List<long>, long>();

        var timeNow = customersQueue.Peek()[0]; // start with first order begin
        long totalWaitingTime = 0;

        while (customersQueue.Count > 0 || ordersTaken.Count > 0)
        {
            while (customersQueue.Count > 0 && customersQueue.Peek()[0] <= timeNow)
            {
                var c = customersQueue.Dequeue();
                ordersTaken.Enqueue(c, c[1]);
            }

            if (ordersTaken.Count > 0)
            {
                var nextOrder = ordersTaken.Dequeue();
                timeNow += nextOrder[1]; // add pizza cooking time
                totalWaitingTime += timeNow - nextOrder[0]; // Вреся ожидания = сейчас минус когда был сделан заказ
            }
            else if (customersQueue.Count > 0) // if there is a gap between customers. т.е. пицца была готова, но новых клиентов еще нет
            {
                timeNow = customersQueue.Peek()[0];
            }
        }

        return totalWaitingTime / customers.Count;
    }

    // Connected Cells in a Grid - HackerRank
    // DFS, Graph, матрица смежности
    // Найти максимальный размер связной компоненты в матрице
    // Идея: затапливаем ячейку, проверяем смежные
    #region Connected Cells in a Grid
    private static int[] dirI = new int[] { -1, -1, -1, 0, 0, 1, 1, 1 };
    private static int[] dirJ = new int[] { 0, -1, 1, -1, 1, 0, -1, 1 };

    public static int connectedCell(List<List<int>> matrix)
    {
        var result = 0;

        for (var i = 0; i < matrix.Count; i++)
        {
            for (var j = 0; j < matrix[i].Count; j++)
            {
                if (matrix[i][j] == 0)
                {
                    continue;
                }

                var size = FindRegionSize(i, j, matrix);

                result = Math.Max(result, size);
            }
        }

        return result;
    }

    private static int FindRegionSize(int ii, int jj, List<List<int>> matrix)
    {
        var result = 0;
        var cells = new Stack<int[]>();
        cells.Push(new int[] { ii, jj });

        while (cells.Count > 0)
        {
            var cell = cells.Pop();
            var i = cell[0];
            var j = cell[1];

            if (i < 0 || i >= matrix.Count)
            {
                continue;
            }
            if (j < 0 || j >= matrix[i].Count)
            {
                continue;
            }
            if (matrix[i][j] == 0)
            {
                continue;
            }

            result++;
            matrix[i][j] = 0;

            for (var d = 0; d < 8; d++)
            {
                cells.Push(new int[] { i + dirI[d], j + dirJ[d] });
            }
        }

        return result;
    }
    #endregion

    // Dijkstra: Shortest Reach 2 - HackerRank
    // Graph, Dijkstra, Priority Queue
    // Найти расстояние до всех вершин графа начиная со стартовой s
    // Учитываются недостижимые вершины
    #region Dijkstra: Shortest Reach 2
    public static List<int> shortestReach(int n, List<List<int>> edges, int s)
    {
        var distance = new int[n];
        var conn = new List<(int, int)>[n];
        var heap = new PriorityQueue<(int, int), int>();

        for (int i = 0; i < n; i++)
        {
            distance[i] = int.MaxValue;
            conn[i] = new List<(int, int)>();
        }

        foreach (var e in edges)
        {
            conn[e[0] - 1].Add((e[1] - 1, e[2]));
            conn[e[1] - 1].Add((e[0] - 1, e[2]));
        }

        distance[s - 1] = 0;
        heap.Enqueue((s - 1, 0), 0);

        while (heap.Count > 0)
        {
            (int town, int townD) = heap.Dequeue();

            if (townD > distance[town])
            {
                continue;
            }

            foreach ((int next, int nextD) in conn[town])
            {
                int newD = townD + nextD;
                if (newD < distance[next])
                {
                    distance[next] = newD;
                    heap.Enqueue((next, distance[next]), distance[next]);
                }
            }
        }

        var result = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (i == s - 1) continue;
            if (distance[i] == int.MaxValue)
            {
                result.Add(-1);
            }
            else
            {
                result.Add(distance[i]);
            }
        }
        return result;
    }
    #endregion
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

        //var arr = new List<int> { 100, 200, 300, 350, 400, 401, 402 };

        //int result = Result.maxMin(3, arr);
        //Console.WriteLine(result);

        //var k = 100000;
        //var num = 105823341;
        var num = 100000;
        var k = 105823341;
        var arr = new List<int>();
        for (int i = 0; i < num; i++)
        {
            arr.Add(1);
        }
        int result = Result.cookies(k, arr);

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