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
