namespace LeetcodePreapare;

// https://leetcode.com/problem-list/r3q9lspc/
public static class LeetCodeBlind75
{
    // #1
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

    // #2
    // 128. Longest Consecutive Sequence
    // HashSet
    // Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
    // O(n) time
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

    // #3
    // 128. Longest Consecutive Sequence
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
}
