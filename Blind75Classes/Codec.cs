namespace LeetcodePreapare;

// LeetCode: 449. Serialize and Deserialize BST
// Serialization is converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer,
// or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
// Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work.
// You need to ensure that a binary search tree can be serialized to a string, and this string can be deserialized to the original tree structure.
// The encoded string should be as compact as possible.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int x) { val = x; }
 * }
 */

// Более оптимальный вариант для BST - сохранять только значения узлов в префиксном порядке (preorder traversal),
// Попробовать такой подход:
// Грубо говоря, обходить дерево в глубину: мы записываем родитель, дальше все значения слева, потом все значения справа
// При чтении, если мы встречаем зачение больше родителя, значит мы переходим к правому поддереву
// Значение null не храним вообще

public class Codec // NOT OPTIMAL, works for all binary trees, for specifically Binary Serach Tree can be better
{

    // Encodes a tree to a single string.
    public string serialize(TreeNode root)
    {
        var vals = new List<string>();

        var queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            if (node is null)
            {
                vals.Add("#");
            }
            else
            {
                vals.Add(node.val.ToString());
                queue.Enqueue(node.left);
                queue.Enqueue(node.right);
            }

        }

        return string.Join(',', vals);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(string data)
    {
        var vals = data.Split(',');
        var n = vals.Length;
        if (n == 0) return null;
        if (vals[0] == "#") return null;

        var queue = new Queue<TreeNode>();
        var root = new TreeNode(int.Parse(vals[0]));
        queue.Enqueue(root);
        var i = 1;
        while (i < n && queue.Count > 0)
        {
            var node = queue.Dequeue();
            var leftVal = vals[i];
            i++;
            var rightVal = i < n ? vals[i] : "#";
            i++;
            if (leftVal != "#")
            {
                node.left = new TreeNode(int.Parse(leftVal));
                queue.Enqueue(node.left);
            }
            if (rightVal != "#")
            {
                node.right = new TreeNode(int.Parse(rightVal));
                queue.Enqueue(node.right);
            }
        }

        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// String tree = ser.serialize(root);
// TreeNode ans = deser.deserialize(tree);
// return ans;
