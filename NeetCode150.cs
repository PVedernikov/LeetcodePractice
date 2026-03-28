namespace LeetcodePreapare;

public class NeetCode150
{
    #region Arrays & Hashing

    // 36. Valid Sudoku
    // Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
    // 1. Each row must contain the digits 1-9 without repetition.
    // 2. Each column must contain the digits 1-9 without repetition.
    // 3. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
    // Note:
    // A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    // Only the filled cells need to be validated according to the mentioned rules.
    #region 36. Valid Sudoku
    public bool IsValidSudoku(char[][] board)
    {
        for (int i = 0; i < 9; i++)
        {
            var seen = new bool[9];
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] == '.') continue;
                var num = (int)(board[i][j] - '1');
                if (seen[num]) return false;
                seen[num] = true;
            }
        }

        for (int i = 0; i < 9; i++)
        {
            var seen = new bool[9];
            for (int j = 0; j < 9; j++)
            {
                if (board[j][i] == '.') continue;
                var num = (int)(board[j][i] - '1');
                if (seen[num]) return false;
                seen[num] = true;
            }
        }

        for (int si = 0; si <= 6; si += 3)
        {
            for (int sj = 0; sj <= 6; sj += 3)
            {
                var seen = new bool[9];
                for (int i = si; i < si + 3; i++)
                {
                    for (int j = sj; j < sj + 3; j++)
                    {
                        if (board[i][j] == '.') continue;
                        var num = (int)(board[i][j] - '1');
                        if (seen[num]) return false;
                        seen[num] = true;
                    }
                }
            }
        }

        return true;
    }
    #endregion

    #endregion

    #region Sliding Window

    // 567. Permutation in String
    // You are given two strings s1 and s2.
    // Return true if s2 contains a permutation of s1, or false otherwise. That means if a permutation of s1 exists as a substring of s2, then return true.
    // Both strings only contain lowercase letters.
    // Note: Permutation is anagram, so using sliding window and counting frequencies
    #region 567. Permutation in String

    // Array
    public bool CheckInclusion(string s1, string s2)
    {
        var n1 = s1.Length;
        var n2 = s2.Length;

        if (n1 > n2) return false;

        var f1 = new int[26];
        var f2 = new int[26];
        var required = 0;
        for (int i = 0; i < n1; i++)
        {
            var c = s1[i] - 'a';
            if (f1[c] == 0) required++;

            f1[c]++;
        }

        var l = 0;
        var r = 0;
        var matched = 0;
        while (r < n2)
        {
            var c = s2[r] - 'a';
            if (f2[c] == f1[c]) matched--;
            f2[c]++;
            if (f2[c] == f1[c]) matched++;
            r++;

            if (r - l > n1)
            {
                var cl = s2[l] - 'a';
                if (f2[cl] == f1[cl]) matched--;
                f2[cl]--;
                if (f2[cl] == f1[cl]) matched++;
                l++;
            }

            if (matched == required) return true;
        }

        return false;
    }

    // Dictionary - same thing, but a little slower because dictinary needs to calculate hash for each key
    public bool CheckInclusionD(string s1, string s2)
    {
        var n1 = s1.Length;
        var n2 = s2.Length;

        if (n1 > n2) return false;

        var f1 = new Dictionary<char, int>();
        var f2 = new Dictionary<char, int>();
        var s1Symbols = 0;
        for (int i = 0; i < n1; i++)
        {
            if (!f1.ContainsKey(s1[i]))
            {
                f1[s1[i]] = 1;
                f2[s1[i]] = 0;
                s1Symbols++;
            }
            else
            {
                f1[s1[i]]++;
            }
        }

        var l = 0;
        var r = 0;
        var matched = 0;
        while (r < n2)
        {
            if (f2.ContainsKey(s2[r]))
            {
                if (f2[s2[r]] == f1[s2[r]]) matched--;

                f2[s2[r]]++;

                if (f2[s2[r]] == f1[s2[r]]) matched++;
            }

            if (matched == s1Symbols) return true;
            r++;

            if (r - l >= n1)
            {
                if (f2.ContainsKey(s2[l]))
                {
                    if (f2[s2[l]] == f1[s2[l]]) matched--;

                    f2[s2[l]]--;

                    if (f2[s2[l]] == f1[s2[l]]) matched++;
                }

                l++;
            }
        }

        return false;
    }
    #endregion


    #endregion


    #region Binary Search

    // 704. Binary Search
    // Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums.
    // If target exists, then return its index. Otherwise, return -1.
    // You must write an algorithm with O(log n) runtime complexity.
    #region 704. Binary Search
    public int Search(int[] nums, int target)
    {
        var l = 0;
        var r = nums.Length - 1;
        while (l <= r)
        {
            // l + (r - l)/2 = l + r/2 - l/2 = l(1 - 1/2) + r/2 = l/2 + r/2 = (l + r) / 2
            var mid = (r + l) / 2; // same as l + (r - l) / 2;
            if (target == nums[mid]) return mid;
            if (target < nums[mid])
            {
                r = mid - 1;
            }
            else
            {
                l = mid + 1;
            }
        }

        return -1;
    }
    #endregion

    #endregion

    #region Trees
    // 543. Diameter of Binary Tree
    // Given the root of a binary tree, return the length of the diameter of the tree.
    // The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
    // The length of a path between two nodes is represented by the number of edges between them.
    #region 543. Diameter of Binary Tree
    public int DiameterOfBinaryTree(TreeNode root)
    {
        if (root is null) return 0;
        var result = 0;
        DiameterOfBinaryTreeDFS(root);
        return result;

        int DiameterOfBinaryTreeDFS(TreeNode root)
        {
            if (root is null) return 0;
            var leftResult = DiameterOfBinaryTreeDFS(root.left);
            var rightResult = DiameterOfBinaryTreeDFS(root.right);

            result = Math.Max(result, leftResult + rightResult);
            return 1 + Math.Max(leftResult, rightResult);
        }
    }
    #endregion
    
    // 1448. Count Good Nodes in Binary Tree
    // Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.
    // Return the number of good nodes in the binary tree.
    #region 1448. Count Good Nodes in Binary Tree
    public int GoodNodes(TreeNode root)
    {
        return GoodNodesDFS(root, int.MinValue);
    }

    private int GoodNodesDFS(TreeNode root, int max)
    {
        if (root is null) return 0;
        var currMax = Math.Max(max, root.val);
        var result = GoodNodesDFS(root.left, currMax) + GoodNodesDFS(root.right, currMax);
        if (root.val >= max) result++;
        return result;
    }
    #endregion
    #endregion

    #region Heap / Priority Queue

    // 1046. Last Stone Weight
    // You are given an array of integers stones where stones[i] is the weight of the ith stone.
    // We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together.
    // Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:
    // - If x == y, both stones are destroyed, and
    // - If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
    // At the end of the game, there is at most one stone left.
    // Return the weight of the last remaining stone. If there are no stones left, return 0.
    #region 1046. Last Stone Weight
    public int LastStoneWeight(int[] stones)
    {
        var heap = new PriorityQueue<int, int>();
        for (int i = 0; i < stones.Length; i++)
        {
            heap.Enqueue(stones[i], -stones[i]);
        }

        while (heap.Count > 1)
        {
            var st1 = heap.Dequeue();
            var st2 = heap.Dequeue();
            if (st1 != st2)
            {
                var resSt = st1 - st2;
                heap.Enqueue(resSt, -resSt);
            }
        }

        return heap.Count > 0 ? heap.Peek() : 0;
    }
    #endregion

    #endregion

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

    // 90. Subsets II
    // Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
    // The solution set must not contain duplicate subsets. Return the solution in any order.
    // Идея: для каждого повторяющегося элемента добавляем его в уже существующие варианты, которые были созданы на предыдущем шаге, и добавляем эти новые варианты в результат.
    // TODO: переписать на каноничное решение: Если число новое, расширяем ВСЕ подмножества. Если дубликат, расширяем только "свежие" подмножества, т.е. только те, которые были добавлены на предыдущем шаге
    #region 90. Subsets II
    public IList<IList<int>> SubsetsWithDup(int[] nums)
    {
        var result = new List<IList<int>>();
        result.Add(new List<int>());
        Array.Sort(nums);
        var i = 0;
        while (i < nums.Length)
        {
            var index = i;
            var newSubsets = new List<IList<int>>();
            var newSubset = new List<int>();
            // Формируем все варианты из повторяющихся элементов, например [2], [2, 2], [2, 2, 2] и т.д.
            while (i < nums.Length && nums[index] == nums[i])
            {
                newSubset.Add(nums[i]);
                newSubsets.Add(new List<int>(newSubset));
                i++;
            }

            var tmp = new List<IList<int>>();
            foreach (var res in result)
            {
                foreach (var sub in newSubsets)
                {
                    var newRes = new List<int>(res);
                    newRes.AddRange(sub);
                    tmp.Add(newRes);
                }
            }

            result.AddRange(tmp);
        }

        return result;
    }
    #endregion

    // 39. Combination Sum
    // Given an array of distinct integers candidates and a target integer target,
    // return a list of all unique combinations of candidates where the chosen numbers sum to target.
    // You may return the combinations in any order.
    // The same number may be chosen from candidates an unlimited number of times.
    // Two combinations are unique if the frequency of at least one of the chosen numbers is different.
    // The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
    #region 39. Combination Sum
    public IList<IList<int>> CombinationSum(int[] candidates, int target)
    {
        IList<IList<int>> result = new List<IList<int>>();
        CSum(0, 0, new List<int>(), candidates, target, result);

        return result;
    }

    private void CSum(
        int i,
        int currSum,
        IList<int> comb,
        int[] candidates,
        int target,
        IList<IList<int>> result)
    {
        if (currSum == target)
        {
            var copyComb = new List<int>(comb);
            result.Add(copyComb);
            return;
        }

        if (i >= candidates.Length || currSum > target) return;

        comb.Add(candidates[i]);
        var newSum = currSum + candidates[i];
        CSum(i, currSum + candidates[i], comb, candidates, target, result);

        comb.RemoveAt(comb.Count - 1);
        CSum(i + 1, currSum, comb, candidates, target, result);
    }
    #endregion

    // 40. Combination Sum II
    // Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
    // Each number in candidates may only be used once in the combination.
    // Note: The solution set must not contain duplicate combinations.
    #region 40. Combination Sum II
    public IList<IList<int>> CombinationSum2(int[] candidates, int target)
    {
        var n = candidates.Length;
        var result = new List<IList<int>>();
        Array.Sort(candidates);
        CSum2(candidates, target, 0, new List<int>(), result);

        return result;
    }

    private void CSum2(int[] candidates, int target, int index, IList<int> curr, IList<IList<int>> result)
    {
        if (index >= candidates.Length) return;
        if (candidates[index] == target)
        {
            var sum = new List<int>(curr);
            sum.Add(candidates[index]);
            result.Add(sum);
            return;
        }

        if (candidates[index] > target)
        {
            return; // since array is sorted, no reason to go 
        }

        var i = index + 1;
        while (i < candidates.Length && candidates[i] == candidates[index])
        {
            i++;
        }

        if (candidates[index] < target)
        {
            var sum = new List<int>(curr);
            sum.Add(candidates[index]);
            CSum2(candidates, target - candidates[index], index + 1, sum, result);
            CSum2(candidates, target, i, curr, result);
        }
    }
    #endregion

    // 46. Permutations
    // Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
    // Вывести все перестановки массива.
    // Идея: для каждого элемента вставляем его во все возможные позиции уже существующих перестановок, например для [1, 2] и элемента 3 получаем [3, 1, 2], [1, 3, 2], [1, 2, 3]
    #region 46. Permutations
    public IList<IList<int>> Permute(int[] nums)
    {
        var n = nums.Length;
        var result = new List<IList<int>>();
        result.Add(new List<int>());
        for (int i = 0; i < n; i++)
        {
            var res = new List<IList<int>>();
            foreach (var mut in result)
            {
                for (int j = 0; j <= mut.Count; j++)
                {
                    var newMut = new List<int>(mut);
                    newMut.Insert(j, nums[i]);
                    res.Add(newMut);
                }
            }
            result = res;
        }

        return result;
    }
    #endregion

    // 22. Generate Parentheses
    // Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    // Input: n = 3
    // Output: ["((()))","(()())","(())()","()(())","()()()"]
    // TODO: рассмотреть классический случай и решить нормально
    #region 22. Generate Parentheses
    public IList<string> GenerateParenthesis(int n)
    {
        return GenerateParenthesisDFS(n, new Dictionary<int, HashSet<string>>()).ToList();
    }

    public HashSet<string> GenerateParenthesisDFS(int n, Dictionary<int, HashSet<string>> cache)
    {
        if (n < 1) return [""];
        if (n == 1) return ["()"];

        if (cache is not null && cache.ContainsKey(n))
        {
            return cache[n];
        }

        var result = new HashSet<string>();

        for (int i = 0; i < n; i++)
        {
            var sl = GenerateParenthesisDFS(i, cache);
            var sr = GenerateParenthesisDFS(n - i - 1, cache);
            foreach (var strl in sl)
            {
                foreach (var strr in sr)
                {
                    result.Add($"{strl}({strr})");
                }
            }
        }

        cache[n] = result;
        return result;
    }
    #endregion


    // 17. Letter Combinations of a Phone Number
    // Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
    // A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
    #region 17. Letter Combinations of a Phone Number
    public IList<string> LetterCombinations(string digits)
    {
        var n = digits.Length;
        var chars = new Dictionary<char, string>
        {
            { '2', "abc" }, { '3', "def" }, { '4', "ghi" }, { '5', "jkl" },
            { '6', "mno" }, { '7', "pqrs" }, { '8', "tuv" }, { '9', "wxyz" },
        };

        var queue = new Queue<string>();
        queue.Enqueue(string.Empty);
        for (int i = 0; i < n; i++)
        {
            var qCount = queue.Count;
            while (qCount > 0)
            {
                var currStr = queue.Dequeue();
                for (int j = 0; j < chars[digits[i]].Length; j++)
                {
                    queue.Enqueue(currStr + chars[digits[i]][j]);
                }
                qCount--;
            }
        }

        var result = new List<string>();
        while (queue.Count > 0)
        {
            result.Add(queue.Dequeue());
        }

        return result;
    }
    #endregion


    #endregion

    #region 1-D Dynamic Programming

    // 746. Min Cost Climbing Stairs
    // You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.
    // You can either start from the step with index 0, or the step with index 1.
    // Return the minimum cost to reach the top of the floor.
    #region 746. Min Cost Climbing Stairs
    public int MinCostClimbingStairs(int[] cost)
    {
        var n = cost.Length;
        var min0 = cost[0];
        var min1 = cost[1];
        for (int i = 2; i < n; i++)
        {
            var minI = cost[i] + Math.Min(min0, min1);
            min0 = min1;
            min1 = minI;
        }

        return Math.Min(min0, min1);
    }
    #endregion
    #endregion

    #region 2-D Dynamic Programming

    // 518. Coin Change II
    // You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
    // Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
    // You may assume that you have an infinite number of each kind of coin.
    // The answer is guaranteed to fit into a signed 32-bit integer.
    #region 518. Coin Change II
    public int Change(int amount, int[] coins)
    {
        var n = coins.Length;
        var dp = new int[n + 1, amount + 1];
        for (int i = 0; i <= n; i++)
        {
            dp[i, 0] = 1;
        }

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= amount; j++)
            {
                if (coins[i - 1] > j)
                {
                    dp[i, j] = dp[i - 1, j];
                }
                else
                {
                    dp[i, j] = dp[i, j - coins[i - 1]] + dp[i - 1, j];
                }
            }
        }

        return dp[n, amount];
    }
    #endregion

    // 72. Edit Distance
    // Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
    // You have the following three operations permitted on a word:
    // - Insert a character
    // - Delete a character
    // - Replace a character
    #region 72. Edit Distance
    public int MinDistance(string word1, string word2)
    {
        var n1 = word1.Length;
        var n2 = word2.Length;
        if (n2 == 0) return n1;
        if (n1 == 0) return n2;

        var dp = new int[n1 + 1, n2 + 1];
        for (int i = 1; i <= n1; i++)
        {
            dp[i, 0] = i;
        }
        for (int j = 1; j <= n2; j++)
        {
            dp[0, j] = j;
        }

        for (int i = 1; i <= n1; i++)
        {
            for (int j = 1; j <= n2; j++)
            {
                if (word1[i - 1] == word2[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1]; // nothing to do
                }
                else
                {
                    int insert = dp[i, j - 1];
                    int delete = dp[i - 1, j];
                    int replace = dp[i - 1, j - 1];

                    dp[i, j] = Math.Min(replace, Math.Min(insert, delete)) + 1;
                }
            }
        }

        return dp[n1, n2];
    }
    #endregion

    #endregion



    #region Greedy


    // 55. Jump Game
    // You are given an integer array nums. You are initially positioned at the array's first index,
    // and each element in the array represents your maximum jump length at that position
    // Return true if you can reach the last index, or false otherwise.
    #region 55. Jump Game
    public bool CanJump(int[] nums)
    {
        var n = nums.Length;
        if (n <= 1) return true;

        var maxIndex = 0;

        for (int i = 0; i <= maxIndex; i++)
        {
            maxIndex = Math.Max(maxIndex, i + nums[i]);
            if (maxIndex >= n - 1) return true;
        }

        return false;
    }
    #endregion


    // 45. Jump Game II
    // You are given a 0-indexed array of integers nums of length n. You are initially positioned at index 0.
    // Each element nums[i] represents the maximum length of a forward jump from index i.
    // In other words, if you are at index i, you can jump to any index (i + j) where:
    // 0 <= j <= nums[i] and
    // i + j < n
    // Return the minimum number of jumps to reach index n - 1. The test cases are generated such that you can reach index n - 1.
    #region 45. Jump Game II

    // Greedy solution
    // O(n) time complexity
    public int Jump(int[] nums)
    {
        var n = nums.Length;
        var maxDistance = 0;
        var start = 0;
        var result = 0;
        while (maxDistance < n - 1)
        {
            var maxFromThisLevel = maxDistance;
            for (int i = start; i <= maxDistance; i++)
            {
                maxFromThisLevel = Math.Max(maxFromThisLevel, i + nums[i]);
            }
            start = maxDistance + 1;
            maxDistance = maxFromThisLevel;
            result++;
        }

        return result;
    }

    // DP solution
    // NOT OPTIMAL
    // O(n^2) time complexity
    public int JumpDP(int[] nums)
    {
        var n = nums.Length;
        var dp = new int[n];

        for (int i = 1; i < n; i++)
        {
            dp[i] = int.MaxValue;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 1; j <= nums[i]; j++)
            {
                if (i + j >= n) break;
                dp[i + j] = Math.Min(dp[i + j], dp[i] + 1);
            }
        }

        return dp[n - 1];
    }
    #endregion

    #endregion

    #region Math & Geometry

    // 50. Pow(x, n)
    // Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).
    // Идея: как бинарный поиск, если n четное, то x^n = (x^2)^(n/2), если n нечетное, то x^n = x * (x^2)^(n/2)
    #region 50. Pow(x, n)
    public double MyPow(double x, int n)
    {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1.0 / x;

        var result = MyPow(x, n / 2);
        result *= result;
        var odd = n % 2;
        if (odd == 1) result *= x;
        if (odd == -1) result /= x;

        return result;
    }
    #endregion

    // 43. Multiply Strings
    // Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
    // Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
    #region 43. Multiply Strings
    public string Multiply(string num1, string num2)
    {
        var n1 = num1.Length;
        var n2 = num2.Length;
        var nm1 = new int[n1];
        var nm2 = new int[n2];
        var rn = n1 + n2;
        var res = new int[rn];
        for (int i = 0; i < n1; i++)
        {
            nm1[n1 - i - 1] = num1[i] - '0';
        }

        for (int i = 0; i < n2; i++)
        {
            nm2[n2 - i - 1] = num2[i] - '0';
        }

        for (int i = 0; i < n2; i++)
        {
            var r = Mul43(nm1, n1, nm2[i]);
            Add43(res, r, i);
        }

        // From here just construct the result
        // Skip leading zeroes
        var zeroesCount = 0;
        while (zeroesCount < rn && res[rn - zeroesCount - 1] == 0)
        {
            zeroesCount++;
        }
        if (zeroesCount == rn) return "0";

        var resultLength = rn - zeroesCount;
        var result = new char[resultLength];
        for (int i = 0; i < resultLength; i++)
        {
            // Reverse digits
            result[resultLength - i - 1] = (char)('0' + res[i]);
        }

        return new string(result);
    }

    private void Add43(int[] result, int[] num, int shift)
    {
        var n = num.Length;
        var carry = 0;
        for (int i = 0; i < n; i++)
        {
            var res = result[i + shift] + num[i] + carry;
            result[i + shift] = res % 10;
            carry = res / 10;
        }

        var ii = n + shift;
        while (carry > 0 && ii < result.Length)
        {
            var res = result[ii] + carry;
            result[ii] = res % 10;
            carry = res / 10;
            ii++;
        }
    }

    private int[] Mul43(int[] num, int n, int x)
    {
        var result = new int[n + 1];
        var carry = 0;
        for (int i = 0; i < n; i++)
        {
            var res = num[i] * x + carry;
            result[i] = res % 10;
            carry = res / 10;
        }

        result[n] = carry;
        return result;
    }
    #endregion

    #endregion
}
