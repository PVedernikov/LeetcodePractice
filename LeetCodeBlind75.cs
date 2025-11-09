using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetcodePreapare;

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


}
