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

    #region Two Pointers

    // 125. Valid Palindrome
    // A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters,
    // it reads the same forward and backward. Alphanumeric characters include letters and numbers.
    // Given a string s, return true if it is a palindrome, or false otherwise.
    // Example 1:
    // Input: s = "A man, a plan, a canal: Panama"
    // Output: true
    // Explanation: "amanaplanacanalpanama" is a palindrome.
    #region 125. Valid Palindrome
    public bool IsPalindrome(string s)
    {
        var l = 0;
        var r = s.Length - 1;
        while (l < r)
        {
            if (!IsValid(s[l]))
            {
                l++;
                continue;
            }

            if (!IsValid(s[r]))
            {
                r--;
                continue;
            }

            var cl = ToLower(s[l]);
            var cr = ToLower(s[r]);
            if (cl != cr) return false;
            l++;
            r--;
        }

        return true;

        bool IsValid(char c)
        {
            return (c >= '0' && c <= '9')
                || (c >= 'a' && c <= 'z')
                || (c >= 'A' && c <= 'Z');
        }

        char ToLower(char c)
        {
            return (c >= 'A' && c <= 'Z')
                ? (char)(c - 'A' + 'a')
                : c;
        }
    }
    #endregion

    // 15. 3Sum
    // Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    // Notice that the solution set must not contain duplicate triplets.
    #region 15. 3Sum
    public IList<IList<int>> ThreeSum(int[] nums)
    {
        Array.Sort(nums);
        var n = nums.Length;

        var result = new List<IList<int>>();
        for (int i = 0; i < n - 2; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1]) continue; // skip duplicates
            var l = i + 1;
            var r = n - 1;
            while (l < r)
            {
                var s = nums[i] + nums[l] + nums[r];
                if (s < 0)
                {
                    l++;
                }
                else if (s > 0)
                {
                    r--;
                }
                else
                {
                    result.Add(new List<int> { nums[i], nums[l], nums[r] });
                    l++;
                    while (l < r && nums[l] == nums[l - 1]) // skip duplicates
                    {
                        l++;
                    }
                    // equal to
                    // r--;
                    // while (l < r && nums[r] == nums[r + 1])
                    // {
                    //     r--;
                    // }
                }
            }
        }

        return result;
    }
    #endregion

    // 42. Trapping Rain Water
    // Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
    // https://leetcode.com/problems/trapping-rain-water/description/
    // HARD, two pointers
    // O(n) / O(1)
    #region 42. Trapping Rain Water
    // Two pointers solution, O(n) / O(1), optimal
    public int Trap(int[] height)
    {
        var n = height.Length;
        var l = 0;
        var r = n - 1;
        var maxL = 0;
        var maxR = 0;
        var result = 0;
        while (l <= r)
        {
            if (maxL < maxR)
            {
                result += Math.Max(0, maxL - height[l]);
                maxL = Math.Max(maxL, height[l]);
                l++;
            }
            else
            {
                result += Math.Max(0, maxR - height[r]);
                maxR = Math.Max(maxR, height[r]);
                r--;
            }
        }

        return result;
    }

    // First attempt
    // O(n) / O(1) - optimal
    // Идея: найти индекс пика, который будет разделять массив на две части. Слева от пика вода будет скапливаться до уровня пика, справа от пика вода будет скапливаться до уровня пика.
    // Например: [0,1,0,2,1,0,1,3,2,1,2,1], Output: 6
    //
    //        #
    //    #~~~##~#
    // _#~##~######
    // 010210132121
    // Проходим по каждой части и считаем количество воды, которая может там скапливаться.
    public int Trap_Peak(int[] height)
    {
        var n = height.Length;
        var max = -1;
        var peak = n - 1;
        for (int i = 0; i < n; i++)
        {
            if (max < height[i])
            {
                max = height[i];
                peak = i;
            }
        }

        var result = 0;
        var prevHeight = 0;
        var localResult = 0;
        for (int i = 0; i <= peak; i++) // Идем слева до пика
        {
            if (prevHeight <= height[i])
            {
                result += localResult;
                localResult = 0;
                prevHeight = height[i];
            }
            else
            {
                localResult += prevHeight - height[i];
            }
        }

        prevHeight = 0;
        localResult = 0;
        for (int i = n - 1; i >= peak; i--) // Идем справа до пика
        {
            if (prevHeight <= height[i])
            {
                result += localResult;
                localResult = 0;
                prevHeight = height[i];
            }
            else
            {
                localResult += prevHeight - height[i];
            }
        }

        return result;
    }

    // finding left and right walls height for each position
    // O(n) / O(n), NOT OPTIMAL for memory
    // Идея: для каждой позиции найти высоту левой стенки и правой стенки, которая может удерживать воду.
    // Высота воды на позиции будет равна min(leftWall, rightWall) - height[i]. Если результат отрицательный, значит вода там скапливаться не будет.
    public int Trap_Walls(int[] height)
    {
        var n = height.Length;
        var leftMax = new int[n];
        var rightMax = new int[n];
        var max = 0;
        for (int i = 0; i < n; i++) // Находим высоту левой стенки для каждой позиции
        {
            leftMax[i] = max;
            max = Math.Max(max, height[i]);
        }
        max = 0;
        for (int i = n - 1; i >= 0; i--) // Находим высоту правой стенки для каждой позиции
        {
            rightMax[i] = max;
            max = Math.Max(max, height[i]);
        }
        var result = 0;
        for (int i = 0; i < n; i++) // Находим количество воды для каждой позиции, которая может там скапливаться, исходя из высот левой и правой стенкок
        {
            var level = Math.Min(leftMax[i], rightMax[i]) - height[i];
            result += level > 0 ? level : 0;
        }

        return result;
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

    // 239. Sliding Window Maximum
    // You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
    // You can only see the k numbers in the window. Each time the sliding window moves right by one position.
    // Return the max sliding window.
    // То есть, нужно вернуть максимум для каждого положения окна

    #region 239. Sliding Window Maximum
    // First attempt, linear search, O(n * k) time, NOT OPTIMAL
    public int[] MaxSlidingWindow_Linear(int[] nums, int k)
    {
        var n = nums.Length;
        var result = new List<int>();
        var maxI = -1;
        var maxVal = int.MinValue;
        for (int i = 0; i < k; i++)
        {
            if (nums[i] >= maxVal)
            {
                maxVal = nums[i];
                maxI = i;
            }
        }

        for (int i = 0; i <= n - k; i++)
        {
            if (nums[i + k - 1] >= maxVal)
            {
                maxVal = nums[i + k - 1];
                maxI = i + k - 1;
            }
            else if (maxI < i)
            {
                maxVal = int.MinValue;
                for (int j = 0; j < k; j++) // this finding max in not optimal
                {
                    if (nums[i + j] >= maxVal)
                    {
                        maxVal = nums[i + j];
                        maxI = i + j;
                    }
                }
            }

            result.Add(maxVal);
        }

        return result.ToArray();
    }

    // Heap solution, O(n log k) time, NOT OPTIMAL
    // Heap, PriorityQueue
    // TODO: implement optimal solution with deque, O(n) time and O(k) space
    public int[] MaxSlidingWindow_Heap(int[] nums, int k)
    {
        var n = nums.Length;
        var result = new List<int>();
        var heap = new PriorityQueue<(int index, int val), int>();
        for (int i = 0; i < k - 1; i++)
        {
            heap.Enqueue((i, nums[i]), -nums[i]);
        }

        for (int i = 0; i <= n - k; i++)
        {
            heap.Enqueue((i + k - 1, nums[i + k - 1]), -nums[i + k - 1]);
            while (heap.Peek().index < i)
            {
                heap.Dequeue();
            }
            result.Add(heap.Peek().val);
        }

        return result.ToArray();
    }

    #endregion

    #endregion

    #region Stack
    // 20. Valid Parentheses
    // Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    // An input string is valid if:
    // 1. Open brackets must be closed by the same type of brackets.
    // 2. Open brackets must be closed in the correct order.
    // 3. Every close bracket has a corresponding open bracket of the same type.
    #region 20. Valid Parentheses
    public bool IsValid(string s)
    {
        var n = s.Length;
        if (n % 2 > 0) return false;
        var stack = new Stack<char>();
        for (int i = 0; i < n; i++)
        {
            if (s[i] == '(' || s[i] == '{' || s[i] == '[')
            {
                stack.Push(s[i]);
            }
            else
            {
                if (stack.Count == 0) return false;
                var bracket = stack.Pop();
                if (bracket == '(' && s[i] != ')') return false;
                if (bracket == '{' && s[i] != '}') return false;
                if (bracket == '[' && s[i] != ']') return false;
            }
        }

        return stack.Count == 0;
    }
    #endregion


    // 155. Min Stack
    // Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
    // Implement the MinStack class:
    // - MinStack() initializes the stack object.
    // - void push(int val) pushes the element val onto the stack.
    // - void pop() removes the element on the top of the stack.
    // - int top() gets the top element of the stack.
    // - int getMin() retrieves the minimum element in the stack.
    // You must implement a solution with O(1) time complexity for each function.
    #region 155. Min Stack
    public class MinStack
    {
        private MinStackNode _node;

        public MinStack() { }

        public void Push(int val)
        {
            var newNode = new MinStackNode();
            newNode.Value = val;
            newNode.Prev = _node;
            newNode.Min = _node is not null && _node.Min < val
                ? _node.Min
                : val;
            _node = newNode;
        }

        public void Pop()
        {
            if (_node is not null)
            {
                _node = _node.Prev;
            }
        }

        public int Top()
        {
            return _node is not null ? _node.Value : int.MinValue; // not necessary, could just return _node.Value because LeetCode guarantees valid inputs
        }

        public int GetMin()
        {
            return _node is not null ? _node.Min : int.MinValue; // not necessary, could just return _node.Value because LeetCode guarantees valid inputs
        }

        class MinStackNode
        {
            public MinStackNode Prev { get; set; }

            public int Value { get; set; }

            public int Min { get; set; }
        }
    }
    #endregion

    // 150. Evaluate Reverse Polish Notation
    // You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.
    // Evaluate the expression. Return an integer that represents the value of the expression.
    // Note that:
    // - The valid operators are '+', '-', '*', and '/'.
    // - Each operand may be an integer or another expression.
    // - The division between two integers always truncates toward zero.
    // - There will not be any division by zero.
    // - The input represents a valid arithmetic expression in a reverse polish notation.
    // - The answer and all the intermediate calculations can be represented in a 32-bit integer.
    #region 150. Evaluate Reverse Polish Notation
    public int EvalRPN(string[] tokens)
    {
        var stack = new Stack<int>();

        for (int i = 0; i < tokens.Length; i++)
        {
            if (tokens[i] == "+")
            {
                var b = stack.Pop();
                var a = stack.Pop();
                stack.Push(a + b);
                continue;
            }
            if (tokens[i] == "-")
            {
                var b = stack.Pop();
                var a = stack.Pop();
                stack.Push(a - b);
                continue;
            }
            if (tokens[i] == "*")
            {
                var b = stack.Pop();
                var a = stack.Pop();
                stack.Push(a * b);
                continue;
            }
            if (tokens[i] == "/")
            {
                var b = stack.Pop();
                var a = stack.Pop();
                stack.Push(a / b);
                continue;
            }
            stack.Push(int.Parse(tokens[i]));
        }
        return stack.Pop();
    }
    #endregion

    // 739. Daily Temperatures
    // Given an array of integers temperatures represents the daily temperatures,
    // return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.
    // If there is no future day for which this is possible, keep answer[i] == 0 instead.
    #region 739. Daily Temperatures
    public int[] DailyTemperatures(int[] temperatures)
    {
        var n = temperatures.Length;
        var result = new int[n];
        var stack = new Stack<int>();

        for (int i = 0; i < n; i++)
        {
            while (stack.Count > 0)
            {
                var j = stack.Peek();
                if (temperatures[j] >= temperatures[i]) break;
                stack.Pop();
                result[j] = i - j;
            }
            stack.Push(i);
        }

        return result;
    }
    #endregion

    // 853. Car Fleet
    // There are n cars at given miles away from the starting mile 0, traveling to reach the mile target.
    // You are given two integer arrays position and speed, both of length n, where position[i] is the starting mile of the ith car and speed[i] is the speed of the ith car in miles per hour.
    // A car cannot pass another car, but it can catch up and then travel next to it at the speed of the slower car.
    // A car fleet is a single car or a group of cars driving next to each other. The speed of the car fleet is the minimum speed of any car in the fleet.
    // If a car catches up to a car fleet at the mile target, it will still be considered as part of the car fleet.
    // Return the number of car fleets that will arrive at the destination.
    #region 853. Car Fleet
    public int CarFleet(int target, int[] position, int[] speed)
    {
        var n = position.Length;
        var cars = new int[n][];
        for (int i = 0; i < n; i++)
        {
            var dist = target - position[i];
            cars[i] = new[] { dist, speed[i] };
        }
        Array.Sort(cars, (a, b) => a[0].CompareTo(b[0]));

        var leadTime = double.MinValue;
        var result = 0;
        for (int i = 0; i < n; i++)
        {
            var dst = cars[i][0];
            var spd = cars[i][1];
            var time = (double)dst / spd;
            if (leadTime < time)
            {
                leadTime = time;
                result++;
            }
        }
        return result;
    }

    // Решение через кучу. По асимптотике вроде тоже самое, но работает медленнее из-за накладных расходов на кучу
    // На LeetCode дает хуже результат
    public int CarFleetHeap(int target, int[] position, int[] speed)
    {
        var n = position.Length;
        var queue = new PriorityQueue<(int, int), int>();
        for (int i = 0; i < n; i++)
        {
            var dist = target - position[i];
            queue.Enqueue((dist, speed[i]), dist);
        }

        var result = 0;
        double leadTime = double.MinValue;
        while (queue.Count > 0)
        {
            (var dst, var spd) = queue.Dequeue();
            var time = ((double)dst) / spd;
            if (leadTime < time)
            {
                leadTime = time;
                result++;
            }
        }
        return result;
    }
    #endregion

    // 84. Largest Rectangle in Histogram
    // Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
    // https://leetcode.com/problems/largest-rectangle-in-histogram
    // Идея: мототонно возрастающий стек. Используем стек, в котором храним пары (индекс, высота). Проходим по массиву высот и для каждой высоты проверяем,
    // есть ли в стеке элементы с большей высотой. Если есть, то удаляем их из стека, т.к. они для нас не актуальны, т.к. с текущей позиции прямоугольник ограничен текущей высотой,
    // В конце рассчитывает площади элементов, которые остались в стеке
    #region 84. Largest Rectangle in Histogram
    public int LargestRectangleArea(int[] heights)
    {
        var stack = new Stack<(int key, int val)>();
        var n = heights.Length;
        var result = 0;
        for (int i = 0; i < n; i++)
        {
            var j = i;
            while (stack.Count > 0 && stack.Peek().val >= heights[i])
            {
                // Удаляем из стека все элементы, которые выше текущего, т.к. они не могут быть частью прямоугольника, т.к. прямоукольник ограничен текущей высотой.
                var h = stack.Pop();
                // Перед удалением рассчитываем площадь всех прямоугольников, которые могли бы получиться с этим элементом.
                // (i - 1) - стартуем с предыдушего индекса, т.к. нужно рассчитать предыдущие прямоугольники, а для текущего элемента расчитаем потом.
                var a = h.val * (i - h.key); // == (i - 1) - h.key + 1
                result = Math.Max(result, a);
                j = h.key;
            }

            stack.Push((j, heights[i])); // Добавляем элемент и позицию, с которой его высота актуальна
        }

        while (stack.Count > 0) // рассчитываем остаток в стеке
        {
            var h = stack.Pop();
            // (n - 1) - стартуем с последнего индекса
            var area = h.val * (n - h.key); // == (n - 1) - h.key + 1
            result = Math.Max(result, area);
        }

        return result;
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

    // 74. Search a 2D Matrix
    // You are given an m x n integer matrix matrix with the following two properties:
    // - Each row is sorted in non-decreasing order.
    // - The first integer of each row is greater than the last integer of the previous row.
    // Given an integer target, return true if target is in matrix or false otherwise.
    // You must write a solution in O(log(m * n)) time complexity.
    #region 74. Search a 2D Matrix
    public bool SearchMatrix(int[][] matrix, int target)
    {
        var m = matrix.Length;
        var n = matrix[0].Length;
        var t = 0;
        var b = m - 1;
        var row = -1;
        while (t <= b)
        {
            var mid = (b + t) / 2;
            if (target >= matrix[mid][0] && target <= matrix[mid][n - 1])
            {
                row = mid;
                break;
            }
            if (target < matrix[mid][0])
            {
                b = mid - 1;
            }
            else
            {
                t = mid + 1;
            }
        }

        if (row == -1) return false; // ранний выход, если не нашли строку, в которой может находиться искомое число 

        var l = 0;
        var r = n - 1;
        while (l <= r)
        {
            var mid = (r + l) / 2;
            if (target == matrix[row][mid]) return true;
            if (target < matrix[row][mid])
            {
                r = mid - 1;
            }
            else
            {
                l = mid + 1;
            }
        }

        return false;
    }
    #endregion

    // 875. Koko Eating Bananas
    // Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.
    // Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile.
    // If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.
    // Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.
    // Return the minimum integer k such that she can eat all the bananas within h hours.
    // Идея: бинарный поиск. Ответ лежит в диапазоне от 1 до max(piles). Вычисляем среуднюю скорость speed =  min + (min - max)/2 = (min + max)/2
    // Далее смотрим сколько часов потребуется, чтобы съесть все бананы с такой скоростью: hours = сумма всех (piles[i]/speed). 
    // если hours > h, значит скорость слишком маленькая, нужно увеличить min до speed + 1. Иначе, если hours <= h, значит скорость может быть меньше, нужно уменьшить max до speed - 1 и запомнить результат как потенциальный ответ.
    #region 875. Koko Eating Bananas
    public int MinEatingSpeed(int[] piles, int h)
    {
        var n = piles.Length;
        var minSpeed = 1;
        var maxSpeed = piles.Max();

        if (n == h) return maxSpeed;

        var result = maxSpeed;
        while (minSpeed <= maxSpeed)
        {
            var speed = (minSpeed + maxSpeed) / 2;
            long hours = 0; // long, иначе на некоторых тестах может быть переполнение
            for (int i = 0; i < n; i++)
            {
                //hours += (long) Math.Ceiling((double)piles[i] / speed);
                hours += (piles[i] + speed - 1) / speed; // standart integer ceil
            }

            if (hours <= h)
            {
                result = speed;
                maxSpeed = speed - 1;
            }
            else
            {
                minSpeed = speed + 1;
            }
        }

        return result;
    }
    #endregion

    // 981. Time Based Key-Value Store
    // Design a time-based key-value data structure that can store multiple values for the same key at different time stamps
    // and retrieve the key's value at a certain timestamp.
    // Implement the TimeMap class:
    // - TimeMap() Initializes the object of the data structure.
    // - void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
    // - String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp.
    //   If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".
    //
    // All the timestamps timestamp of set are strictly increasing.
    // Важный момент: в условии сказано, что вызовы set будут идти в порядке возрастания timestamp, поэтому можно вставлять прямо в начало/конец.
    // Expected complexity: O(log n) for get and O(1) for set.
    // TODO: implement optimal solution with binary search
    #region 981. Time Based Key-Value Store
    // First attempt, Not optimal solution: set O(n), get O(n)
    public class TimeMap
    {
        private Dictionary<string, ListNode> dict;
        public TimeMap()
        {
            dict = new Dictionary<string, ListNode>();
        }

        public void Set(string key, string value, int timestamp)
        {
            var val = new ListNode
            {
                k = key,
                v = value,
                t = timestamp
            };

            if (!dict.ContainsKey(key))
            {
                dict[key] = val;
            }
            else
            {
                if (dict[key].t <= timestamp)
                {
                    val.next = dict[key];
                    dict[key] = val;
                }
                else
                {
                    var curr = dict[key];
                    while (curr.next is not null && curr.next.t > val.t)
                    {
                        curr = curr.next;
                    }
                    val.next = curr.next;
                    curr.next = val;
                }
            }
        }

        public string Get(string key, int timestamp)
        {
            if (!dict.ContainsKey(key))
                return string.Empty;
            var curr = dict[key];
            while (curr is not null && curr.t > timestamp)
            {
                curr = curr.next;
            }

            return curr is null ? string.Empty : curr.v;
        }

        class ListNode
        {
            public string k { get; set; }
            public string v { get; set; }
            public int t { get; set; }
            public ListNode next { get; set; }
        }
    }
    #endregion

    // 4. Median of Two Sorted Arrays
    // Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
    // The overall run time complexity should be O(log (m+n)).
    // HARD
    // Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
    #region 4. Median of Two Sorted Arrays
    // Linear solution, O(m + n) time and O(m + n) space
    // NOT OPTIMAL
    // TODO: implement optimal solution with binary search, O(log(min(m, n))) time and O(1) space
    public double FindMedianSortedArraysLinear(int[] nums1, int[] nums2)
    {
        var m = nums1.Length;
        var n = nums2.Length;
        if (n == 0 || m == 0)
        {
            return n == 0
                ? GetMedian(nums1)
                : GetMedian(nums2);
        }

        var merged = new int[m + n];
        var i = 0;
        var j = 0;
        var curr = 0;
        while (i < m && j < n)
        {
            if (nums1[i] < nums2[j])
            {
                merged[curr] = nums1[i];
                i++;
            }
            else
            {
                merged[curr] = nums2[j];
                j++;
            }
            curr++;
        }

        while (i < m)
        {
            merged[curr] = nums1[i];
            i++;
            curr++;
        }

        while (j < n)
        {
            merged[curr] = nums2[j];
            j++;
            curr++;
        }

        return GetMedian(merged);

        double GetMedian(int[] arr)
        {
            var len = arr.Length;
            if (len == 0)
                return double.MinValue;

            return len % 2 > 0
                ? (double)arr[len / 2]
                : ((double)arr[len / 2] + (double)arr[len / 2 - 1]) / 2;
        }
    }

    // Linear solution, O(m + n) time and O(m + n) space, but optimized for early exit, если мы уже нашли медиану, то не нужно продолжать сливать массивы
    // NOT OPTIMAL
    public double FindMedianSortedArraysLinearOptimized(int[] nums1, int[] nums2)
    {
        var m = nums1.Length;
        var n = nums2.Length;
        if (n == 0 || m == 0)
        {
            return n == 0
                ? GetMedian(nums1)
                : GetMedian(nums2);
        }

        var mid = (m + n) / 2;
        var merged = new int[mid + 1]; // Нам не нужен фулл саайз массив для мержа, нам нужно только до медианы
        var i = 0;
        var j = 0;
        var curr = 0;
        while (i < m && j < n && curr <= mid)
        {
            if (nums1[i] < nums2[j])
            {
                merged[curr] = nums1[i];
                i++;
            }
            else
            {
                merged[curr] = nums2[j];
                j++;
            }
            curr++;
        }

        // Если curr < mid + 1, значит либо nums1, либо nums2 закончился раньше, просто доливаем оставшиеся элементы
        while (curr <= mid && i < m)
        {
            merged[curr] = nums1[i];
            i++;
            curr++;
        }

        while (curr <= mid && j < n)
        {
            merged[curr] = nums2[j];
            j++;
            curr++;
        }

        return (m + n) % 2 > 0
            ? (double)merged[mid]
            : ((double)merged[mid] + (double)merged[mid - 1]) / 2;


        double GetMedian(int[] arr)
        {
            var len = arr.Length;
            if (len == 0)
                return double.MinValue;

            return len % 2 > 0
                ? (double)arr[len / 2]
                : ((double)arr[len / 2] + (double)arr[len / 2 - 1]) / 2;
        }
    }
    #endregion
    #endregion

    #region Linked List

    // 138. Copy List with Random Pointer
    // A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
    // Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node.
    // Both the next and random pointer of the new nodes should point to new nodes in the copied list such that
    // the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.
    // For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.
    // Return the head of the copied linked list.
    // The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:
    // - val: an integer representing Node.val
    // - random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
    // Your code will only be given the head of the original linked list.
    #region 138. Copy List with Random Pointer
    public Node138 CopyRandomList(Node138 head)
    {
        if (head is null) return null;

        var cache = new Dictionary<Node138, Node138>();
        var curr = head;
        while (curr is not null)
        {
            cache[curr] = new Node138(curr.val);
            curr = curr.next;
        }

        curr = head;
        while (curr is not null)
        {
            var newNode = cache[curr];
            var nextNode = curr.next is null
                ? null
                : cache[curr.next];
            var randomNode = curr.random is null
                ? null
                : cache[curr.random];
            newNode.next = nextNode;
            newNode.random = randomNode;
            curr = curr.next;
        }
        return cache[head];
    }

    public class Node138
    {
        public int val;
        public Node138 next;
        public Node138 random;

        public Node138(int _val)
        {
            val = _val;
            next = null;
            random = null;
        }
    }
    #endregion

    // 287. Find the Duplicate Number
    // Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
    // There is only one repeated number in nums, return this repeated number.
    // You must solve the problem without modifying the array nums and using only CONSTANT extra space.
    // TODO: implement optimal solution with cycle detection (Floyd's Tortoise and Hare)
    #region 287. Find the Duplicate Number
    // O(n) memory, but fast
    public int FindDuplicate(int[] nums)
    {
        var n = nums.Length;
        var visited = new bool[n];

        for (int i = 0; i < n; i++)
        {
            if (visited[nums[i] - 1])
                return nums[i];
            visited[nums[i] - 1] = true;
        }

        return 0;
    }

    // O(1) memory, but slow
    // public int FindDuplicate(int[] nums)
    // {
    //     var n = nums.Length;
    //     Array.Sort(nums);

    //     for (int i = 1; i < n; i++)
    //     {
    //         if(nums[i] == nums[i - 1]) return nums[i];
    //     }

    //     return 0;
    // }
    #endregion

    // 146. LRU Cache
    // Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    // Implement the LRUCache class:
    // - LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    // - int get(int key) Return the value of the key if the key exists, otherwise return -1.
    // - void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache.
    //   If the number of keys exceeds the capacity from this operation, evict the least recently used key.
    // The functions get and put must each run in O(1) average time complexity.
    // TODO: переписать на каноничное решение, с dummy head и tail, чтобы не проверять на null и не делать лишних проверок в методах Remove и Insert
    // TODO: перепичать для capacity = 0
    #region 146. LRU Cache
    public class LRUCache
    {
        private int _capacity = 0;
        private Dictionary<int, Node> _nodes;
        private Node _head;

        public LRUCache(int capacity)
        {
            _capacity = capacity;
            _nodes = new Dictionary<int, Node>();
        }

        public int Get(int key)
        {
            if (_nodes.ContainsKey(key))
            {
                var node = _nodes[key];
                Remove(node);
                Insert(node);
                return _nodes[key].Val;
            }

            return -1;
        }

        public void Put(int key, int value)
        {
            Node node;
            if (!_nodes.ContainsKey(key))
            {
                node = new Node();
                node.Key = key;
                _nodes[key] = node;
            }
            else
            {
                node = _nodes[key];
                Remove(node);
            }

            node.Val = value;
            Insert(node);

            if (_nodes.Count > _capacity)
            {
                var last = _head.Prev;
                _nodes.Remove(last.Key);
                last.Prev.Next = _head;
                _head.Prev = last.Prev;
            }
        }
        private void Remove(Node node)
        {
            if (node == _head)
            {
                return;
            }

            node.Prev.Next = node.Next;
            node.Next.Prev = node.Prev;
        }

        private void Insert(Node node)
        {
            if (node == _head)
            {
                return;
            }

            if (_head is null)
            {
                _head = node;
                _head.Next = _head;
                _head.Prev = _head;
            }
            else
            {
                node.Next = _head;
                node.Prev = _head.Prev;
                _head.Prev.Next = node;
                _head.Prev = node;
                _head = node;
            }
        }

        class Node
        {
            public int Val { get; set; }
            public int Key { get; set; }
            public Node Next { get; set; }
            public Node Prev { get; set; }
        }
    }

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache obj = new LRUCache(capacity);
     * int param_1 = obj.Get(key);
     * obj.Put(key,value);
     */
    #endregion

    // 25. Reverse Nodes in k-Group
    // Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
    // k is a positive integer and is less than or equal to the length of the linked list.
    // If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
    // You may not alter the values in the list's nodes, only nodes themselves may be changed.
    // https://leetcode.com/problems/reverse-nodes-in-k-group/
    // Example 1:
    // Input: head = [1,2,3,4,5], k = 2
    // Output: [2,1,4,3,5]
    // Example 2:
    // Input: head = [1,2,3,4,5], k = 3
    // Output: [3,2,1,4,5]
    // Развернуть элементы группами по k элементов. Если в конце осталось меньше k элементов, то их не разворачиваем.
    #region 25. Reverse Nodes in k-Group
    public ListNode25 ReverseKGroup(ListNode25 head, int k)
    {
        if (k <= 1) return head;
        var c = head;
        var groups = 0;
        while (c is not null)
        {
            groups++;
            c = c.next;
        }
        groups /= k; // Считаем количество полных групп по k элементов, т.к. хвост не нужен разворачивать, если в нем меньше k элементов
        if (groups == 0) return head;

        var dummy = new ListNode25(-1, head);
        var prev = dummy;
        var start = head;
        for (int i = 0; i < groups; i++)
        {
            var curr = start;
            var next = start.next;
            for (int j = 0; j < k - 1; j++)
            {
                var tmp = next.next;
                next.next = curr;
                curr = next;
                next = tmp;
            }
            prev.next = curr;
            start.next = next;

            prev = start;
            start = next;
        }

        return dummy.next;
    }

    public class ListNode25
    {
        public int val;
        public ListNode25 next;
        public ListNode25(int val = 0, ListNode25 next = null)
        {
            this.val = val;
            this.next = next;
        }
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

    // 199. Binary Tree Right Side View
    // Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
    // Examples: https://leetcode.com/problems/binary-tree-right-side-view/description/
    // Идея: обход дерева по уровням (BFS), на каждом уровне добавляем в результат последний элемент, который мы видим, т.е. последний элемент в очереди на этом уровне
    #region 199. Binary Tree Right Side View
    public IList<int> RightSideView(TreeNode root)
    {
        var result = new List<int>();
        if (root is null) return result;

        var queue = new Queue<TreeNode>();

        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var levelCount = queue.Count;
            for (int i = 0; i < levelCount; i++)
            {
                var node = queue.Dequeue();
                if (node.left is not null)
                    queue.Enqueue(node.left);
                if (node.right is not null)
                    queue.Enqueue(node.right);
                if (i == levelCount - 1)
                    result.Add(node.val);
            }
        }

        return result;
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

    // 703. Kth Largest Element in a Stream
    // You are part of a university admissions office and need to keep track of the kth highest test score from applicants in real-time.
    // This helps to determine cut-off marks for interviews and admissions dynamically as new applicants submit their scores.
    // You are tasked to implement a class which, for a given integer k, maintains a stream of test scores and continuously returns the kth highest test score after a new score has been submitted.
    // More specifically, we are looking for the kth highest score in the sorted list of all scores.
    // Implement the KthLargest class:
    // - KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of test scores nums.
    // - int add(int val) Adds a new test score val to the stream and returns the element representing the kth largest element in the pool of test scores so far.
    #region 703. Kth Largest Element in a Stream
    public class KthLargest
    {
        private int _k;
        private PriorityQueue<int, int> _minQueue;
        public KthLargest(int k, int[] nums)
        {
            _k = k;
            _minQueue = new PriorityQueue<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                Add(nums[i]);
            }
        }

        public int Add(int val)
        {
            if (_minQueue.Count < _k)
            {
                _minQueue.Enqueue(val, val);
            }
            else if (_minQueue.Peek() < val)
            {
                _minQueue.Dequeue();
                _minQueue.Enqueue(val, val);
            }

            return _minQueue.Peek();
        }
    }

    /**
     * Your KthLargest object will be instantiated and called as such:
     * KthLargest obj = new KthLargest(k, nums);
     * int param_1 = obj.Add(val);
     */
    #endregion

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

    // 973. K Closest Points to Origin
    // Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).
    // The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)^2 + (y1 - y2)^2).
    // You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).
    #region 973. K Closest Points to Origin
    public int[][] KClosest(int[][] points, int k)
    {
        var n = points.Length;
        var result = new int[k][];
        var heap = new PriorityQueue<int[], long>();
        for (int i = 0; i < n; i++)
        {
            var point = points[i];
            long x = point[0]; // long чтобы избежать переполнения при возведении в квадрат, например для точки [10000, 10000] будет 10000^2 + 10000^2 = 200000000, что уже не помещается в int
            long y = point[1]; // -//-
            long distance = x * x + y * y;
            heap.Enqueue(point, -distance);

            if (heap.Count > k)
            {
                heap.Dequeue();
            }
        }

        for (int i = 0; i < k; i++)
        {
            result[i] = heap.Dequeue();
        }

        return result;
    }
    #endregion

    // 215. Kth Largest Element in an Array
    // Given an integer array nums and an integer k, return the kth largest element in the array.
    // Note that it is the kth largest element in the sorted order, not the kth distinct element.
    // Can you solve it without sorting?
    // Идея: кладем элементы в мин-кучу, если размер кучи больше k, удаляем минимальный элемент.
    // В итоге в куче останется k наибольших элементов, а минимальный элемент в куче будет k-ым по величине.
    #region 215. Kth Largest Element in an Array
    public int FindKthLargest(int[] nums, int k)
    {
        var n = nums.Length;
        var queue = new PriorityQueue<int, int>();
        for (int i = 0; i < n; i++)
        {
            queue.Enqueue(nums[i], nums[i]);
            if (queue.Count > k)
            {
                queue.Dequeue();
            }
        }

        return queue.Peek();
    }
    #endregion

    // 621. Task Scheduler
    #region 621. Task Scheduler
    // TODO 

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

    // 131. Palindrome Partitioning
    // Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
    // Example 1:
    // Input: s = "aab"
    // Output: [["a", "a", "b"],["aa", "b"]]
    #region 131. Palindrome Partitioning
    public IList<IList<string>> Partition(string s)
    {
        var n = s.Length;
        var result = new List<IList<string>>();
        var stack = new Stack<string>();

        DFS(0);
        return result;

        void DFS(int i)
        {
            if (i > n - 1)
            {
                var part = new List<string>(stack);
                part.Reverse();
                result.Add(part);
                return;
            }

            for (int j = i; j < n; j++)
            {
                var isPalindrome = true; // TODO: подумать как оптимизировать проверку на палиндром
                var ii = i;
                var jj = j;
                while (ii < jj)
                {
                    if (s[ii] != s[jj])
                    {
                        isPalindrome = false;
                        break;
                    }
                    ii++;
                    jj--;
                }

                if (isPalindrome)
                {
                    stack.Push(s.Substring(i, j - i + 1));
                    DFS(j + 1);
                    stack.Pop();
                }
            }
        }
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

    // 51. N-Queens
    // The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
    // Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.
    // Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.
    // Complexity: O(n!) time complexity, O(n) space complexity
    #region 51. N-Queens
    public IList<IList<string>> SolveNQueens(int n)
    {
        var cols = new HashSet<int>();
        var d1 = new HashSet<int>();
        var d2 = new HashSet<int>();
        var row = new char[n];
        for (int i = 0; i < n; i++)
        {
            row[i] = '.';
        }
        var lines = new string[n]; // init all possible rows with queens for easy use
        for (int i = 0; i < n; i++) 
        {
            row[i] = 'Q'; // i-th row has queen at i-th position
            lines[i] = new string(row);
            row[i] = '.';
        }

        var board = new List<string>();
        var result = new List<IList<string>>();
        QueensBacktrack(0);
        return result;

        void QueensBacktrack(int i)
        {
            if (i >= n)
            {
                result.Add(new List<string>(board)); // new List to create a copy of the board
                return;
            }

            for (int j = 0; j < n; j++)
            {
                if (cols.Contains(j) || d1.Contains(i - j) || d2.Contains(i + j))
                {
                    continue;
                }

                cols.Add(j);
                d1.Add(i - j);
                d2.Add(i + j);
                board.Add(lines[j]);

                QueensBacktrack(i + 1);

                // Cleanup the board, remove last row. We can do that using i, because the board here has exactly i+1 rows.
                board.RemoveAt(i); // Maybe more readable is: board.RemoveAt(board.Count - 1)
                cols.Remove(j);
                d1.Remove(i - j);
                d2.Remove(i + j);
            }
        }
    }

    // First attempt, O(n!) time complexity, O(n) space complexity
    // Board is cobstructed in reverse order, i.e. last row is added first, which is not very intuitive
    // But mathematically it is correct, because the solutions should be the same in the end
    public IList<IList<string>> SolveNQueens_1st(int n)
    {
        var cols = new HashSet<int>();
        var d1 = new HashSet<int>();
        var d2 = new HashSet<int>();
        var line = new char[n];
        for (int i = 0; i < n; i++)
        {
            line[i] = '.';
        }
        var lines = new string[n];
        for (int i = 0; i < n; i++)
        {
            line[i] = 'Q';
            lines[i] = new string(line);
            line[i] = '.';
        }

        var ret = new List<IList<string>>();
        var res0 = Queens(0);
        foreach (var r0 in res0)
        {
            if (r0.Count == n)
            {
                ret.Add(r0);
            }
        }
        return ret;

        IList<IList<string>> Queens(int i)
        {
            var result = new List<IList<string>>();
            if (i >= n)
            {
                result.Add(new List<string>());
                return result;
            }

            for (int j = 0; j < n; j++)
            {
                if (cols.Contains(j) || d1.Contains(i - j) || d2.Contains(i + j))
                {
                    continue;
                }
                cols.Add(j);
                d1.Add(i - j);
                d2.Add(i + j);

                var res = Queens(i + 1);
                foreach (var r in res)
                {
                    r.Add(lines[j]);
                    result.Add(r);
                }
                cols.Remove(j);
                d1.Remove(i - j);
                d2.Remove(i + j);
            }

            return result;
        }
    }
    #endregion
    #endregion

    #region Graphs
    // 695. Max Area of Island
    // You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.)
    // You may assume all four edges of the grid are surrounded by water.
    // The area of an island is the number of cells with a value 1 in the island.
    // Return the maximum area of an island in grid. If there is no island, return 0.
    #region 695. Max Area of Island
    // optimal solution, O(m*n) time complexity, O(1) space complexity
    // TODO: get rid of recursion and implement with stack
    public int MaxAreaOfIsland(int[][] grid)
    {
        var m = grid.Length;
        var n = grid[0].Length;
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };

        var result = 0;

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result = Math.Max(result, DFS695(i, j));
            }
        }

        return result;

        int DFS695(int i, int j)
        {
            if (i < 0 || j < 0 || i >= m || j >= n || grid[i][j] != 1)
            {
                return 0;
            }

            var res = 1;
            grid[i][j] = -1;
            for (int k = 0; k < 4; k++)
            {
                res += DFS695(i + di[k], j + dj[k]);
            }

            return res;
        }
    }
    #endregion

    // 286. Walls and Gates
    // You are given an m x n grid rooms initialized with these three possible values.
    // 1) -1 A wall or an obstacle.
    // 2) 0 A gate.
    // 3) INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
    // Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
    // BFS (DFS doesn't work! Точнее работает, но гораздо медленнее)
    #region 286. Walls and Gates
    public void WallsAndGates(int[][] rooms)
    {
        var m = rooms.Length;
        var n = rooms[0].Length;
        var gates = new Queue<(int, int)>();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (rooms[i][j] == 0)
                {
                    gates.Enqueue((i, j));
                }
            }
        }

        var ddi = new int[] { 1, -1, 0, 0 };
        var ddj = new int[] { 0, 0, 1, -1 };
        while (gates.Count > 0)
        {
            (var i, var j) = gates.Dequeue();

            for (int k = 0; k < 4; k++)
            {
                var di = i + ddi[k];
                var dj = j + ddj[k];
                if (di < 0 || dj < 0 || di >= m || dj >= n || rooms[di][dj] != int.MaxValue) continue;
                // if (rooms[di][dj] < rooms[i][j] + 1) continue; // лишнее, т.к. BFS гарантирует, что мы всегда будем идти от меньшего к большему
                // TODO: разобраться почему. 
                rooms[di][dj] = rooms[i][j] + 1;
                gates.Enqueue((di, dj));
            }
        }
    }
    #endregion

    // 130. Surrounded Regions
    // You are given an m x n matrix board containing letters 'X' and 'O', capture regions that are surrounded:
    // - Connect: A cell is connected to adjacent cells horizontally or vertically.
    // - Region: To form a region connect every 'O' cell.
    // Surround: A region is surrounded if none of the 'O' cells in that region are on the edge of the board. Such regions are completely enclosed by 'X' cells.
    // To capture a surrounded region, replace all 'O's with 'X's in-place within the original board. You do not need to return anything.
    // Идея: идем от границ, если встречаем 'O', то запускаем DFS и помечаем все связанные 'O' как '1'.
    // В итоге все 'O', которые не были помечены, это те, которые нужно заменить на 'X', а все '1' нужно вернуть обратно в 'O'
    #region 130. Surrounded Regions
    // Runtime 1 ms Beats 100.00%
    public void Solve(char[][] board)
    {
        var m = board.Length;
        var n = board[0].Length;
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };

        for (int i = 0; i < m; i++)
        {
            DFS130(i, 0);
            DFS130(i, n - 1);
        }

        for (int j = 0; j < n; j++)
        {
            DFS130(0, j);
            DFS130(m - 1, j);
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (board[i][j] == 'O')
                {
                    board[i][j] = 'X';
                }
            }
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (board[i][j] == '1')
                {
                    board[i][j] = 'O';
                }
            }
        }

        void DFS130(int i, int j)
        {
            if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O')
                return;

            board[i][j] = '1';

            for (int k = 0; k < 4; k++)
            {
                DFS130(i + di[k], j + dj[k]);
            }
        }
    }
    #endregion

    // 207. Course Schedule
    // There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
    // You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
    // - For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    // Return true if you can finish all courses. Otherwise, return false.
    #region 207. Course Schedule
    public bool CanFinish(int numCourses, int[][] prerequisites)
    {
        var adj = new List<int>[numCourses];
        for (int i = 0; i < prerequisites.Length; i++)
        {
            var a = prerequisites[i][0];
            var b = prerequisites[i][1];
            if (adj[a] is null)
            {
                adj[a] = new List<int>();
            }
            adj[a].Add(b);
        }

        var finished = new int[numCourses]; // 0 - not processed, 1 - visited, 2 - finished
        for (int i = 0; i < numCourses; i++)
        {
            if (finished[i] == 2) continue;

            if (!CanFinish(i))
            {
                return false;
            }
        }

        return true;

        bool CanFinish(int i)
        {
            if (finished[i] == 2) return true;
            if (finished[i] == 1) return false;

            finished[i] = 1;
            if (adj[i] is not null && adj[i].Count > 0)
            {
                foreach (int b in adj[i])
                {
                    if (!CanFinish(b)) return false;
                }
            }
            finished[i] = 2;
            return true;
        }
    }
    #endregion

    // 210. Course Schedule II
    // There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
    // You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
    // - For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    // Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.
    // Topological Sort, DFS, Graph
    // Time complexity: O(V + E), where V is the number of courses and E is the number of prerequisites
    #region 210. Course Schedule II
    public int[] FindOrder_NotOptimal(int numCourses, int[][] prerequisites)
    {
        var adj = new List<int>[numCourses];
        for (int i = 0; i < prerequisites.Length; i++)
        {
            var a = prerequisites[i][0];
            var b = prerequisites[i][1];
            if (adj[a] is null)
            {
                adj[a] = new List<int>();
            }

            adj[a].Add(b);
        }

        var index = 0;
        var result = new int[numCourses]; 
        var finished = new bool[numCourses];
        for (int i = 0; i < numCourses; i++)
        {
            if (finished[i]) continue;

            if (!CanFinish(i, new bool[numCourses]))
            {
                return [];
            }

        }

        return result;

        bool CanFinish(int a, bool[] visited)
        {
            // TODO
            // Вместо двух массивов visited и finished можно было бы использовать один массив, где 0 - не обработан, 1 - посещали, 2 - закончили
            if (finished[a]) return true;
            if (visited[a]) return false;

            visited[a] = true;
            if (adj[a] is not null && adj[a].Count > 0)
            {
                foreach (int b in adj[a])
                {
                    if (!CanFinish(b, visited))
                    {
                        return false;
                    }
                }
            }

            finished[a] = true;
            result[index] = a;
            index++;
            return true;
        }
    }

    // MORE OPTIMAL
    public int[] FindOrder(int numCourses, int[][] prerequisites)
    {
        var adj = new List<int>[numCourses];
        for (int i = 0; i < prerequisites.Length; i++)
        {
            var a = prerequisites[i][0];
            var b = prerequisites[i][1];
            if (adj[a] is null)
            {
                adj[a] = new List<int>();
            }

            adj[a].Add(b);
        }

        var result = new List<int>(numCourses); // Pass numCourses because size is known, and in this case we avoid unnecessary re-allocations
        var finished = new int[numCourses]; // 0 - not processed, 1 - visited, 2 - finished
        for (int i = 0; i < numCourses; i++)
        {
            if (finished[i] == 2) continue;

            if (!CanFinish(i))
            {
                return [];
            }
        }

        return result.ToArray();

        bool CanFinish(int a)
        {
            if (finished[a] == 2) return true;
            if (finished[a] == 1) return false;

            finished[a] = 1;
            if (adj[a] is not null && adj[a].Count > 0)
            {
                foreach (int b in adj[a])
                {
                    if (!CanFinish(b))
                    {
                        return false;
                    }
                }
            }

            finished[a] = 2;
            result.Add(a);
            return true;
        }
    }
    #endregion

    // 684. Redundant Connection
    // In this problem, a tree is an undirected graph that is connected and has no cycles.
    // You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added.
    // The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed.
    // The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.
    // Return an edge that can be removed so that the resulting graph is a tree of n nodes.
    // If there are multiple answers, return the answer that occurs last in the input.
    #region 684. Redundant Connection

    // first attempt, O(n^2) time complexity, O(n) space complexity
    // NOT OPTIMAL,
    // TODO: implement optimal solution
    // Идея: используем массив unionFind, где unionFind[i] - это родитель вершины i. Изначально каждый элемент является своим родителем.
    // Мы схлопываем весь путь до одной вершитны, т.е. все вершины, которые принадлежат одной компоненте связности, будут иметь одного родителя.
    // Изначально это не так, но когда мы обрабатываем ребра, мы объединяем компоненты связности.
    // Когда мы обрабатываем ребро (a, b), если unionFind[a] == unionFind[b], значит a и b уже в одной компоненте связности, и это ребро лишнее, возвращаем его.
    // Иначе, объединяем компоненты связности, т.е. все вершины, у которых родитель unionFind[b], теперь будут иметь родителя unionFind[a].
    public int[] FindRedundantConnection(int[][] edges)
    {
        var n = edges.Length;
        var unionFind = new int[n + 1];
        for (int i = 1; i <= n; i++)
        {
            unionFind[i] = i;
        }

        for (int i = 0; i < n; i++)
        {
            var a = edges[i][0];
            var b = edges[i][1];

            if (unionFind[a] == unionFind[b]) return edges[i];
            var parentA = unionFind[a];
            var parentB = unionFind[b];
            for (int j = 1; j <= n; j++) // This loop is O(n), and it makes the overall time complexity O(n^2)
            {
                if (unionFind[j] == parentB)
                {
                    unionFind[j] = parentA;
                }
            }
        }

        return null;
    }
    #endregion

    // 127. Word Ladder
    // A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
    // - Every adjacent pair of words differs by a single letter.
    // - Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
    // - sk == endWord
    // Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord,
    // or 0 if no such sequence exists.
    //
    // Example: Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    // Output: 5
    // Explanation: One shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog", which is 5 words long.
    #region 127. Word Ladder
    // First attempt. Naive solution + Dijkstra
    // NOT OPTIMAL, TODO: implement optimal solution with BFS + оптимизировать построение графа через промежуточный adj list
    // где ключ - это маска слова с замененной буквой на *, а значение - это список слов, которые подходят под эту маску.
    // Например для слова "hot" мы будем иметь маски "*ot", "h*t", "ho*", и для каждой маски будем хранить список слов, которые подходят под эту маску.
    // Тогда при построении графа мы будем перебирать все слова, для каждого слова генерировать его маски и добавлять ребра между всеми словами, которые подходят под эти маски.
    // Это позволит нам построить граф за O(n * m^2), где n - количество слов в wordList, m - длина слова, вместо O(n^2 * m) в текущей реализации.
    // Это оптимальнее, т.к. по условию m гораздо меньше n.
    public int LadderLength(string beginWord, string endWord, IList<string> wordList)
    {
        var targetI = -1;
        var n = wordList.Count;

        if (isEarlyExit()) return 0;

        var adj = new List<int>[n + 1];
        var dist = new int[n + 1];

        for (int i = 0; i < n + 1; i++)
        {
            adj[i] = new List<int>();
            dist[i] = int.MaxValue;
        }
        dist[0] = 1;

        for (int i = 0; i < n; i++)
        {
            //if (endWord == wordList[i]) targetI = i + 1;
            if (Connected(beginWord, wordList[i]))
            {
                adj[0].Add(i + 1);
                adj[i + 1].Add(0);
            }

            for (var j = i + 1; j < n; j++)
            {
                if (Connected(wordList[i], wordList[j]))
                {
                    adj[i + 1].Add(j + 1);
                    adj[j + 1].Add(i + 1);
                }
            }
        }

        //if (targetI < 0) return 0;

        var queue = new PriorityQueue<int, int>();
        queue.Enqueue(0, 0);
        while (queue.Count > 0)
        {
            var i = queue.Dequeue();
            foreach (var j in adj[i])
            {
                var dst = dist[i] + 1;
                if (dist[j] > dst)
                {
                    dist[j] = dst;
                    queue.Enqueue(j, dst);
                }
            }
        }

        return targetI > 0 && dist[targetI] != int.MaxValue ? dist[targetI] : 0;

        bool Connected(string s1, string s2)
        {
            var diff = 0;
            for (int i = 0; i < s1.Length; i++)
            {
                if (s1[i] != s2[i]) diff++;
                if (diff > 1) return false;
            }
            return diff == 1;
        }

        bool isEarlyExit()
        {
            for (var i = 0; i < n; i++)
            {
                if (wordList[i] == endWord)
                {
                    targetI = i + 1;
                    return false;
                }
            }
            return true;
        }
    }
    #endregion

    #endregion

    #region Advanced Graphs

    // 743. Network Delay Time
    // You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi),
    // where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.
    // We will send a signal from a given node k.
    // Return the minimum time it takes for all the n nodes to receive the signal.
    // If it is impossible for all the n nodes to receive the signal, return -1.
    // Dijkstra's Algorithm, Graph, Shortest Path
    #region 743. Network Delay Time
    public int NetworkDelayTime(int[][] times, int n, int k)
    {
        var adj = new List<int[]>[n];
        for (int i = 0; i < times.Length; i++)
        {
            var u = times[i][0] - 1;
            var v = times[i][1] - 1;
            var w = times[i][2];

            if (adj[u] is null)
                adj[u] = new List<int[]>();

            adj[u].Add(new int[] { v, w });
        }
        var dist = new int[n];
        for (int i = 0; i < n; i++)
        {
            dist[i] = int.MaxValue;
        }

        var queue = new PriorityQueue<int, int>(); // Dijkstra
        //var queue = new Queue<int>();
        dist[k - 1] = 0;
        queue.Enqueue(k - 1, 0); // Dijkstra
        //queue.Enqueue(k - 1);
        while (queue.Count > 0)
        {
            var i = queue.Dequeue();
            if (adj[i] is null) continue;

            foreach (var conn in adj[i])
            {
                var j = conn[0];
                var speed = dist[i] + conn[1];
                if (dist[j] > speed)
                {
                    dist[j] = speed;
                    queue.Enqueue(j, speed); // Dijkstra
                    //queue.Enqueue(j);
                }
            }
        }

        var result = -1;
        for (int i = 0; i < n; i++)
        {
            if (dist[i] == int.MaxValue)
                return -1;

            result = Math.Max(result, dist[i]);
        }

        return result;
    }
    #endregion

    // 1584. Min Cost to Connect All Points
    // You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].
    // The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.
    // Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.
    // 
    // Prim's Algorithm, Prim's (MST), Minimum Spanning Tree, MST
    // Идея: начинаем с первой точки, добавляем все ребра от нее до остальных точек в мин-кучу,
    // затем на каждом шаге достаем из кучи ребро с минимальным весом, если оно ведет в новую точку,
    // то добавляем эту точку в результат и добавляем все ребра от этой точки до остальных точек в кучу. Повторяем, пока не добавим все точки.
    // Time complexity: O(n^2 log n)
    #region 1584. Min Cost to Connect All Points
    public int MinCostConnectPoints(int[][] points)
    {
        var n = points.Length;
        var result = 0;
        var visited = new bool[n];
        visited[0] = true;
        var heap = new PriorityQueue<(int, int), int>();
        for (int j = 1; j < n; j++)
        {
            var dist = GetDistance(0, j);
            heap.Enqueue((j, dist), dist);
        }
        var count = 1; // Только для раннего выхода, т.к. в таком графе может быть максимум n-1 ребер
        while (heap.Count > 0 && count < n)
        {
            (var i, var dst) = heap.Dequeue();
            if (visited[i]) continue;
            visited[i] = true;
            count++;
            result += dst;
            for (int j = 0; j < n; j++)
            {
                if (i == j || visited[j]) continue;
                var dist = GetDistance(i, j);
                heap.Enqueue((j, dist), dist);
            }
        }

        return result;

        int GetDistance(int i, int j)
        {
            return Math.Abs(points[i][0] - points[j][0])
                    + Math.Abs(points[i][1] - points[j][1]);
        }
    }
    #endregion

    // 787. Cheapest Flights Within K Stops
    // There are n cities connected by some number of flights.
    // You are given an array flights where flights[i] = [from_i, to_i, price_i] indicates that there is a flight from city from_i to city to_i with cost price_i.
    // You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.
    // Dijkstra's Algorithm, Graph, Shortest Path
    #region 787. Cheapest Flights Within K Stops
    public int FindCheapestPrice(int n, int[][] flights, int src, int dst, int k)
    {
        var adj = new List<(int, int)>[n];
        for (int i = 0; i < flights.Length; i++)
        {
            var f = flights[i][0];
            var t = flights[i][1];
            var p = flights[i][2];
            if (adj[f] is null)
                adj[f] = new List<(int, int)>();
            adj[f].Add((t, p));
        }

        var dist = new int[n, k + 2];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k + 2; j++)
            {
                dist[i, j] = int.MaxValue;
            }
        }

        for (int j = 0; j < k + 2; j++)
        {
            dist[src, j] = 0;
        }

        var heap = new PriorityQueue<(int, int), int>();
        heap.Enqueue((src, 0), 0);

        while (heap.Count > 0)
        {
            (var i, var stops) = heap.Dequeue();
            if (stops > k || adj[i] is null) continue;
            foreach ((var j, var distance) in adj[i])
            {
                var d = distance + dist[i, stops];
                if (dist[j, stops + 1] > d)
                {
                    dist[j, stops + 1] = d;
                    if (j != dst)
                    {
                        heap.Enqueue((j, stops + 1), d);
                    }
                }
            }
        }

        var result = int.MaxValue;
        for (int j = 0; j < k + 2; j++)
        {
            result = Math.Min(dist[dst, j], result);
        }
        return result == int.MaxValue ? -1 : result;
    }
    #endregion

    #endregion

    #region 1-D Dynamic Programming

    // 70. Climbing Stairs
    // You are climbing a staircase. It takes n steps to reach the top.
    // Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    #region 70. Climbing Stairs
    public int ClimbStairs(int n)
    {
        var prev1 = 1;
        var prev2 = 0;
        var result = 0;
        for (int i = 1; i <= n; i++)
        {
            result = prev1 + prev2;
            prev2 = prev1;
            prev1 = result;
        }
        return result;
    }

    // memoization solution, O(n) time complexity, O(n) space complexity
    // NOT OPTIMAL for memory
    public int ClimbStairs_memo(int n)
    {
        var dp = new int[n + 1];
        return Climb(n);

        int Climb(int num)
        {
            if (num <= 1) return 1;
            if (num == 2) return 2;
            if (dp[num] > 0) return dp[num];

            var result = Climb(num - 1) + Climb(num - 2);
            dp[num] = result;
            return result;
        }
    }
    #endregion

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
    // 416. Partition Equal Subset Sum
    // Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.
    #region 416. Partition Equal Subset Sum
    // NOT OPTIMAL
    // Решить с использованием реального DP, а не рекурсии с мемоизацией
    public bool CanPartition(int[] nums)
    {
        var n = nums.Length;
        if (n == 1) return false;
        var sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += nums[i];
        }

        if (sum % 2 > 0) return false;
        sum /= 2;

        var cache = new bool?[n, sum + 1];

        return IsPossible(0, sum);

        bool IsPossible(int i, int s)
        {
            if (s == 0) return true;
            if (s < 0) return false;
            if (i >= n) return false;
            if (cache[i, s].HasValue) return cache[i, s] == true;
            cache[i, s] = IsPossible(i + 1, s) || IsPossible(i + 1, s - nums[i]);
            return cache[i, s].Value;
        }
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

    // 494. Target Sum
    // You are given an integer array nums and an integer target.
    // You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
    // For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
    // Return the number of different expressions that you can build, which evaluates to target.
    // Важно: перед 0 можно поставить как +, так и -, и это будет считаться разными выражениями, т.е. для nums = [0] и target = 0 ответ будет 2, т.к. можно построить выражения "+0" и "-0"
    #region 494. Target Sum
    // Naive solution, NOT OPTIMAL
    // TODO: implement 2-D DP solution
    public int FindTargetSumWays(int[] nums, int target)
    {
        var n = nums.Length;
        return Ways(0, target);

        int Ways(int i, int sum)
        {
            if (i == n - 1)
            {
                var result = 0;
                if (sum == nums[i]) result++;
                if (sum == -nums[i]) result++;
                return result;
            }

            return Ways(i + 1, sum + nums[i]) + Ways(i + 1, sum - nums[i]);
        }
    }
    #endregion

    // 97. Interleaving String
    // Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
    // An interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:
    // - s = s1 + s2 + ... + sn
    // - t = t1 + t2 + ... + tm
    // - |n - m| <= 1
    // - The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
    // Note: a + b is the concatenation of strings a and b.
    #region 97. Interleaving String
    public bool IsInterleave(string s1, string s2, string s3)
    {
        var n1 = s1.Length;
        var n2 = s2.Length;
        var n3 = s3.Length;
        if (n3 != n1 + n2) return false;

        var dp = new bool[n1 + 1, n2 + 1];
        dp[0, 0] = true;
        for (int i = 1; i <= n1; i++)
        {
            if (s1[i - 1] != s3[i - 1]) break;
            dp[i, 0] = true;
        }
        for (int j = 1; j <= n2; j++)
        {
            if (s2[j - 1] != s3[j - 1]) break;
            dp[0, j] = true;
        }

        for (int i = 1; i <= n1; i++)
        {
            for (int j = 1; j <= n2; j++)
            {
                dp[i, j] = (s1[i - 1] == s3[i + j - 1] && dp[i - 1, j])
                    || (s2[j - 1] == s3[i + j - 1] && dp[i, j - 1]);
            }
        }

        return dp[n1, n2];
    }
    #endregion


    // 329. Longest Increasing Path in a Matrix
    // Given an m x n integers matrix, return the length of the longest increasing path in matrix.
    // From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).
    // Идея: DFS + мемоизация, т.е. для каждой ячейки запоминаем длину максимального пути, который начинается с этой ячейки.
    #region 329. Longest Increasing Path in a Matrix
    public int LongestIncreasingPath(int[][] matrix)
    {
        var m = matrix.Length;
        var n = matrix[0].Length;
        var di = new int[] { 1, -1, 0, 0 };
        var dj = new int[] { 0, 0, 1, -1 };
        var path = new int[m, n];
        var result = 0;

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result = Math.Max(result, DFS(i, j));
            }
        }

        return result;

        int DFS(int i, int j)
        {
            if (path[i, j] > 0) return path[i, j];
            var res = 0;
            for (int d = 0; d < 4; d++)
            {
                var ii = i + di[d];
                var jj = j + dj[d];
                if (ii < 0 || jj < 0 || ii >= m || jj >= n) continue;
                if (matrix[i][j] >= matrix[ii][jj]) continue;
                res = Math.Max(res, DFS(ii, jj));
            }
            path[i, j] = res + 1;
            return path[i, j];
        }
    }
    #endregion

    // 115. Distinct Subsequences
    // Given two strings s and t, return the number of distinct subsequences of s which equals t.
    // The test cases are generated so that the answer fits on a 32-bit signed integer.
    // Complexity: O(m * n), where m is the length of s and n is the length of t
    // 2d DP, DFS + memoization
    #region 115. Distinct Subsequences
    public int NumDistinct(string s, string t)
    {
        var m = s.Length;
        var n = t.Length;
        var cache = new int[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                cache[i, j] = -1;
            }
        }

        return DFS(0, 0);

        int DFS(int i, int j)
        {
            if (j >= n) return 1;
            if (i >= m) return 0;
            if (cache[i, j] >= 0) return cache[i, j];
            if (m - i < n - j) // early exit, if remaining length of s is less than remaining length of t, then it's impossible to match
            {
                cache[i, j] = 0;
                return 0;
            }

            var result = DFS(i + 1, j); // don't include symbol
            if (s[i] == t[j])
            {
                result += DFS(i + 1, j + 1); // include symbol
            }
            cache[i, j] = result;
            return result;
        }
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

    // 312. Burst Balloons
    // You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. You are asked to burst all the balloons.
    // If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins.
    // If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.
    // Return the maximum coins you can collect by bursting the balloons wisely.
    // HARD
    // 2D DP, DFS
    #region 312. Burst Balloons
    // Идея: дополняем массив nums двумя единицами в начале и в конце, чтобы упростить вычисление.
    // Инвариант: для каждого элемента nums[i] мы считаем, что он последний, т.е. лопаем сначала все элементы слева и справа от него. 
    // Кешируем результат для каждого диапазона [l, r]. Мы так можем сделать, т.к. элементы слева и справа от диапазона остаются неизменными.
    // Потому что в нашей логике элементы слева и справа от диапазона [l, r] лопаются после самого диапазона, а значит не меняются.
    public int MaxCoins(int[] nums)
    {
        var n = nums.Length;
        var eNums = new int[n + 2];
        eNums[0] = 1;
        eNums[n + 1] = 1;
        for (int i = 1; i < n + 1; i++)
        {
            eNums[i] = nums[i - 1];
        }
        var dp = new int[n + 2, n + 2];
        for (int i = 0; i < n + 2; i++)
        {
            for (int j = 0; j < n + 2; j++)
            {
                dp[i, j] = -1;
            }
        }

        return Coins(1, n);

        int Coins(int l, int r)
        {
            if (l > r) return 0;
            if (dp[l, r] >= 0) return dp[l, r];

            var result = 0;
            for (int i = l; i <= r; i++)
            {
                var lRes = Coins(l, i - 1);
                var rRes = Coins(i + 1, r);
                var res = eNums[i] * eNums[l - 1] * eNums[r + 1] + lRes + rRes;
                result = Math.Max(result, res);
            }

            dp[l, r] = result;
            return result;
        }
    }
    #endregion

    // 10. Regular Expression Matching
    // Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
    // - '.' Matches any single character.​​​​
    // - '*' Matches zero or more of the preceding element.
    // Return a boolean indicating whether the matching covers the entire input string (not partial).
    // HARD
    // 2D DP
    #region 10. Regular Expression Matching
    public bool IsMatch(string s, string p)
    {
        var m = s.Length;
        var n = p.Length;
        var dp = new bool[m + 1, n + 1];
        dp[0, 0] = true;

        // Инициализируем dp[0, j], т.е. случаи, когда строка s пустая, а паттерн p может быть не пустой.
        // В этом случае, паттерн может совпадать со строкой только если он состоит из пар символов, где второй символ - '*', т.е. может означать 0 вхождений первого символа.
        // Например, для паттерна "a*b*c*" строка "" будет совпадать, т.к. '*' может означать 0 вхождений
        for (int j = 2; j <= n; j++)
        {
            if (p[j - 1] == '*')
            {
                dp[0, j] = dp[0, j - 2];
            }
        }

        for (int i = 1; i < m + 1; i++)
        {
            for (var j = 1; j < n + 1; j++)
            {
                if (s[i - 1] == p[j - 1] || p[j - 1] == '.')
                {
                    dp[i, j] = dp[i - 1, j - 1]; // если символы совпадают (или '.'), то результат такой же, как для строк без этих символов
                }
                else if (p[j - 1] == '*')
                {
                    // если встретили '*', то два варианта:
                    // 1) использовать символ p[j - 2] (т.е. тот, который перед '*'), тогда он должен совпадать с текущим символом строки s[i - 1] (или это '.')
                    //    + должно быть так, что строка s до этого символа уже совпадала с паттерном, который включает '*', т.е. dp[i - 1, j] должно быть true,
                    //    например проверяем baaaa и ba*, т.е. dp[5, 3], тогда для последнего символа строки 'a' мы можем использовать '*' для совпадения с ним,
                    //    и при этом строка до этого символа 'baaa' уже совпадала с паттерном 'ba*', т.е. dp[4, 3] должно быть true
                    // 2) не использовать символ p[j - 2] (т.е. тот, который перед '*'), тогда результат такой же, как для паттерна без p[j - 2]* (т.е. dp[i, j - 2])
                    dp[i, j] = ((s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1, j])
                    || dp[i, j - 2];
                }
                else
                {
                    dp[i, j] = false;
                }
            }
        }

        return dp[m, n];
    }

    // NeetCode recursive solution with memoization
    public bool IsMatchRecursive(string s, string p)
    {
        var m = s.Length;
        var n = p.Length;
        var cache = new Dictionary<(int, int), bool>();
        return DFS(0, 0);

        bool DFS(int i, int j)
        {
            if (cache.ContainsKey((i, j))) return cache[(i, j)];
            if (i >= m && j >= n) return true;
            if (j >= n) return false;

            var match = i < m && (s[i] == p[j] || p[j] == '.');
            if (j + 1 < n && p[j + 1] == '*')
            {
                var result = DFS(i, j + 2) // don't use '*'
                    || (match && DFS(i + 1, j)); // use '*'
                cache[(i, j)] = result;
                return result;
            }
            if (match)
            {
                var result = DFS(i + 1, j + 1);
                cache[(i, j)] = result;
                return result;
            }
            cache[(i, j)] = false;
            return false;
        }
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

    // 134. Gas Station
    // There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
    // You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station.
    // You begin the journey with an empty tank at one of the gas stations.
    // Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction,
    // otherwise return -1. If there exists a solution, it is guaranteed to be unique.
    #region 134. Gas Station
    public int CanCompleteCircuit(int[] gas, int[] cost)
    {
        var n = gas.Length;
        var totalGas = 0;
        var totalCost = 0;
        for (int i = 0; i < n; i++)
        {
            totalGas += gas[i];
            totalCost += cost[i];
        }
        if (totalCost > totalGas) return -1;

        var tank = 0;
        var start = -1;
        for (int i = 0; i < n; i++)
        {
            tank += gas[i] - cost[i];
            if (tank < 0)
            {
                tank = 0;
                start = -1;
            }
            else if (start < 0)
            {
                start = i;
            }
        }

        return start;
    }

    // public int CanCompleteCircuit(int[] gas, int[] cost)
    // {
    //     var n = gas.Length;
    //     for (int start = 0; start < n; start++)
    //     {
    //         if (gas[start] < cost[start])
    //             continue;

    //         var i = start;
    //         var tank = gas[start];
    //         while (true)
    //         {
    //             tank -= cost[i];
    //             i = (i + 1) % n;
    //             if (tank < 0)
    //                 break;
    //             if (i == start)
    //                 return start;
    //             tank += gas[i];
    //         }
    //     }

    //     return -1;
    // }
    #endregion

    // 846. Hand of Straights
    // Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size groupSize, and consists of groupSize consecutive cards.
    // Given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize, return true if she can rearrange the cards, or false otherwise.
    #region 846. Hand of Straights
    // NOT OPTIMAL
    // TODO: implement optimal solution (frequency count + min or max heap)
    public bool IsNStraightHand(int[] hand, int groupSize)
    {
        var n = hand.Length;
        if (n % groupSize > 0) return false;
        if (groupSize == 1) return true;

        var gn = n / groupSize;

        Array.Sort(hand);
        var groups = new int[gn];
        var sizes = new int[gn];
        //var heap = new ProirityQueue<int, int>();
        for (int j = 0; j < gn; j++)
        {
            groups[j] = -1;
            //heap.Enqueue(j, int,)
        }
        for (int i = 0; i < n; i++)
        {
            var found = false;
            // first try to continue one of the groups
            for (int j = 0; j < gn; j++) // optimize this
            {
                if (groups[j] == hand[i] - 1 && sizes[j] < groupSize)
                {
                    groups[j] = hand[i];
                    sizes[j]++;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                // if we can't continue one of the groups, try to start new one
                for (int j = 0; j < gn; j++) // optimize this
                {
                    if (groups[j] == -1)
                    {
                        groups[j] = hand[i];
                        sizes[j]++;
                        found = true;
                        break;
                    }
                }
            }

            if (!found) return false;
        }

        return true;
    }
    #endregion

    // 1899. Merge Triplets to Form Target Triplet
    // A triplet is an array of three integers. You are given a 2D integer array triplets, where triplets[i] = [ai, bi, ci] describes the ith triplet.
    // You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.
    // To obtain target, you may apply the following operation on triplets any number of times (possibly zero):
    // - Choose two indices (0-indexed) i and j (i != j) and update triplets[j] to become [max(ai, aj), max(bi, bj), max(ci, cj)].
    //   - For example, if triplets[i] = [2, 5, 3] and triplets[j] = [1, 7, 5], triplets[j] will be updated to [max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5].
    // Return true if it is possible to obtain the target triplet [x, y, z] as an element of triplets, or false otherwise.
    #region 1899. Merge Triplets to Form Target Triplet
    public bool MergeTriplets(int[][] triplets, int[] target)
    {
        var n = triplets.Length;

        var a = false;
        var b = false;
        var c = false;

        for (int i = 0; i < n; i++)
        {
            var t = triplets[i];
            if (t[0] > target[0] || t[1] > target[1] || t[2] > target[2])
                continue;

            if (t[0] == target[0]) a = true;
            if (t[1] == target[1]) b = true;
            if (t[2] == target[2]) c = true;

            if (a && b && c) return true;
        }

        return false;
    }
    #endregion

    // 763. Partition Labels
    // You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.
    // For example, the string "ababcc" can be partitioned into ["abab", "cc"], but partitions such as ["aba", "bcc"] or ["ab", "ab", "cc"] are invalid.
    // Note that the partition is done so that after concatenating all the parts in order, the resultant string should be s.
    // Return a list of integers representing the size of these parts.
    #region 763. Partition Labels
    // TODO: решение перегружено, нужно упростить. Можно убрать PriorityQueue. Идея, хранить только последнюю позицию, а интервалы строить на лету, когда идем по строке.
    public IList<int> PartitionLabels(string s)
    {
        var n = s.Length;

        var start = new int[26];
        var end = new int[26];
        for (int i = 0; i < 26; i++)
        {
            start[i] = -1;
            end[i] = -1;
        }

        for (int i = 0; i < n; i++)
        {
            var c = s[i] - 'a';
            if (start[c] < 0)
            {
                start[c] = i;
            }

            end[c] = i;
        }

        var queue = new PriorityQueue<(int, int), int>();
        for (int i = 0; i < 26; i++)
        {
            if (start[i] < 0) continue;
            queue.Enqueue((start[i], end[i]), start[i]);
        }

        var result = new List<int>();
        if (queue.Count == 0) return result;
        if (queue.Count == 1)
        {
            result.Add(n);
            return result;
        }

        (var l, var r) = queue.Dequeue();
        while (queue.Count > 0)
        {
            (var l1, var r1) = queue.Dequeue();
            if (l1 > r)
            {
                result.Add(r - l + 1);
                l = l1;
                r = r1;
            }
            else
            {
                r = Math.Max(r, r1);
            }
        }
        result.Add(r - l + 1);

        return result;
    }
    #endregion

    // 678. Valid Parenthesis String
    // Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
    // The following rules define a valid string:
    // - Any left parenthesis '(' must have a corresponding right parenthesis ')'.
    // - Any right parenthesis ')' must have a corresponding left parenthesis '('.
    // - Left parenthesis '(' must go before the corresponding right parenthesis ')'.
    // - '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
    #region 678. Valid Parenthesis String
    // Greedy solution
    // TODO: investigate
    public bool CheckValidString(string s)
    {
        var n = s.Length;
        var leftMin = 0;
        var leftMax = 0;
        for (int i = 0; i < n; i++)
        {
            if (s[i] == '(')
            {
                leftMin++;
                leftMax++;
            }

            if (s[i] == ')')
            {
                leftMin--;
                leftMax--;
            }

            if (s[i] == '*')
            {
                leftMin--;
                leftMax++;
            }

            if (leftMax < 0)
                return false;

            if (leftMin < 0)
                leftMin = 0;
        }

        return leftMin == 0;
    }
    #endregion

    #endregion

    #region Intervals

    // 1851. Minimum Interval to Include Each Query
    // HARD
    // You are given a 2D integer array intervals, where intervals[i] = [lefti, righti] describes the ith interval starting at lefti and ending at righti (inclusive).
    // The size of an interval is defined as the number of integers it contains, or more formally righti - lefti + 1.
    // You are also given an integer array queries. The answer to the jth query is the size of the smallest interval i such that lefti <= queries[j] <= righti.
    // If no such interval exists, the answer is -1.
    // Return an array containing the answers to the queries.
    // Example 1:
    // Input: intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]
    // Output: [3,3,1,4]
    // Идея: сортируем интервалы по левому краю, сортируем запросы, используем PriorityQueue для хранения интервалов, которые могут покрывать текущий запрос.
    // В PriorityQueue храним интервалы по размеру (сначала меньшие), а также по правому краю.
    // Для каждого запроса добавляем в PriorityQueue все интервалы, которые начинаются до или в точке запроса,
    // и удаляем из PriorityQueue все интервалы, которые заканчиваются до точки запроса.
    // Если PriorityQueue не пустой, то верхний элемент - это минимальный интервал, который покрывает запрос.
    #region 1851. Minimum Interval to Include Each Query
    #region with custom comparator
    public int[] MinInterval(int[][] intervals, int[] queries)
    {
        var m = intervals.Length;
        var n = queries.Length;
        Array.Sort(intervals, (a, b) => {
            return a[0].CompareTo(b[0]); // Интервалы сортируем по левому краю
        });
        var qrs = new (int index, int val)[n]; // Нужен, чтобы потом восстановить индексы запросов после сортировки
        for (int k = 0; k < n; k++)
        {
            qrs[k] = (k, queries[k]);
        }
        Array.Sort(qrs, (a, b) => {
            return a.val.CompareTo(b.val); // Сортируем запросы по значению
        });

        // PriorityQueue хранит интервалы по размеру (сначала меньшие), Если размер совпадает, то раньше лежит тот, который заканчивается раньше
        var heap = new PriorityQueue<int[], int[]>(new IntervalComparer());
        var result = new int[n];
        for (int k = 0; k < n; k++)
        {
            result[k] = -1;
        }

        var i = 0;
        for (int j = 0; j < n; j++)
        {
            while (i < m && intervals[i][0] <= qrs[j].val) // Добавляем в PriorityQueue все интервалы, которые начинаются до или в точке запроса
            {
                heap.Enqueue(intervals[i], intervals[i]);
                i++;
            }

            while (heap.Count > 0 && heap.Peek()[1] < qrs[j].val) // Удаляем из PriorityQueue все интервалы, которые заканчиваются до точки запроса
            {
                heap.Dequeue();
            }

            if (heap.Count > 0) // Если PriorityQueue не пустой, то верхний элемент - это минимальный интервал, который покрывает запрос
            {
                var intrvl = heap.Peek();
                result[qrs[j].index] = intrvl[1] - intrvl[0] + 1;
            }
        }

        return result;
    }

    class IntervalComparer : IComparer<int[]> // Класс для сравнения интервалов по размеру, если размер совпадает, то по правому краю
    {
        public int Compare(int[] a, int[] b)
        {
            var aLen = a[1] - a[0] + 1;
            var bLen = b[1] - b[0] + 1;
            if (aLen != bLen)
                return aLen.CompareTo(bLen);
            return a[1].CompareTo(b[1]);
        }
    }
    #endregion
    
    public int[] MinInterval_NoCustomComparer(int[][] intervals, int[] queries)
    {
        var m = intervals.Length;
        var n = queries.Length;
        Array.Sort(intervals, (a, b) => {
            return a[0].CompareTo(b[0]); // Интервалы сортируем по левому краю
        });
        var qrs = new (int index, int val)[n]; // Нужен, чтобы потом восстановить индексы запросов после сортировки
        for (int k = 0; k < n; k++)
        {
            qrs[k] = (k, queries[k]);
        }
        Array.Sort(qrs, (a, b) => {
            return a.val.CompareTo(b.val); // Сортируем запросы по значению
        });

        var heap = new PriorityQueue<int[], int>();
        var result = new int[n];
        for (int k = 0; k < n; k++)
        {
            result[k] = -1;
        }

        var i = 0;
        for (int j = 0; j < n; j++)
        {
            while (i < m && intervals[i][0] <= qrs[j].val) // Добавляем в PriorityQueue все интервалы, которые начинаются до или в точке запроса
            {
                heap.Enqueue(intervals[i], intervals[i][1] - intervals[i][0] + 1);
                i++;
            }

            while (heap.Count > 0 && heap.Peek()[1] < qrs[j].val) // Удаляем из PriorityQueue все интервалы, которые заканчиваются до точки запроса
            {
                heap.Dequeue();
            }

            if (heap.Count > 0) // Если PriorityQueue не пустой, то верхний элемент - это минимальный интервал, который покрывает запрос
            {
                var intrvl = heap.Peek();
                result[qrs[j].index] = intrvl[1] - intrvl[0] + 1;
            }
        }

        return result;
    }

    #endregion

    #endregion

    #region Math & Geometry

    // 202. Happy Number
    // Write an algorithm to determine if a number n is happy.
    // A happy number is a number defined by the following process:
    // - Starting with any positive integer, replace the number by the sum of the squares of its digits.
    // - Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    // - Those numbers for which this process ends in 1 are happy.
    // Return true if n is a happy number, and false if not.
    #region 202. Happy Number
    public bool IsHappy(int n)
    {
        var cache = new HashSet<int>();
        while (!cache.Contains(n))
        {
            if (n == 1)
                return true;

            cache.Add(n);

            var newN = 0;
            while (n > 0)
            {
                var d = n % 10;
                n /= 10;
                newN += d * d;
            }

            n = newN;
        }
        return false;
    }
    #endregion

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
