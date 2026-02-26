namespace LeetcodePreapare;

public class NeetCode150
{
    // Backtracking
    #region Backtracking
    // 78. Subsets
    // Given an integer array nums of unique elements, return all possible subsets (the power set).
    // The solution set must not contain duplicate subsets. Return the solution in any order.
    // Идея: просто перебераем все варианты, для каждого элемента добавляем его в уже существующие варианты и добавляем эти новые варианты в результат
    // O(2^n) time complexity
    #region 78. Subsets
    public IList<IList<int>> Subsets(int[] nums)
    {
        var result = new List<IList<int>>();
        result.Add(new List<int>());
        for (int i = 0; i < nums.Length; i++)
        {
            var subResult = new List<IList<int>>();
            foreach (var subset in result)
            {
                var newSubset = new List<int>(subset);
                newSubset.Add(nums[i]);
                subResult.Add(newSubset);
            }
            result.AddRange(subResult);
        }

        return result;
    }
    #endregion

    #endregion
}
