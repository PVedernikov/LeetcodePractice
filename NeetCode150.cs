namespace LeetcodePreapare;

public class NeetCode150
{
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
    #endregion
}
