namespace LeetcodePreapare;

// Blind 75, NeetCode 150
// https://leetcode.com/problem-list/r3q9lspc/
// 208. Implement Trie (Prefix Tree)
// A prefix tree (also known as a trie) is a tree data structure used to efficiently store and retrieve keys in a set of strings.
// Some applications of this data structure include auto-complete and spell checker systems.
// Implement the PrefixTree class:
// - PrefixTree() Initializes the prefix tree object.
// - void insert(String word) Inserts the string word into the prefix tree.
// - boolean search(String word) Returns true if the string word is in the prefix tree (i.e., was inserted before), and false otherwise.
// - boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
#region 208. Implement Trie (Prefix Tree)
// this is good solution, use it.
public class PrefixTree
{
    private PrefixTree[] chars = new PrefixTree[26];

    private bool isEnd = false;

    public PrefixTree() { }

    public void Insert(string word)
    {
        var current = this;
        var n = word.Length;
        for (int i = 0; i < n; i++)
        {
            var j = word[i] - 'a';
            if (current.chars[j] is null) current.chars[j] = new PrefixTree();
            current = current.chars[j];
        }
        current.isEnd = true;
    }

    public bool Search(string word)
    {
        var current = this;
        var n = word.Length;
        for (int i = 0; i < n; i++)
        {
            var j = word[i] - 'a';
            if (current.chars[j] is null) return false;
            current = current.chars[j];
        }
        return current.isEnd;
    }

    public bool StartsWith(string prefix)
    {
        var current = this;
        var n = prefix.Length;
        for (int i = 0; i < n; i++)
        {
            var j = prefix[i] - 'a';
            if (current.chars[j] is null) return false;
            current = current.chars[j];
        }
        return true;
    }
}
#endregion

// Blind 75 
// https://leetcode.com/problem-list/r3q9lspc/
// #51
// 208. Implement Trie (Prefix Tree)
// A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings.
// There are various applications of this data structure, such as autocomplete and spellchecker.
// Implement the Trie class:
// - Trie() Initializes the trie object.
// - void insert(String word) Inserts the string word into the trie.
// - boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
// - boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

public class Trie
{
    private TrieNode[] roots;

    public Trie()
    {
        roots = new TrieNode[26];
    }

    public void Insert(string word)
    {
        var nodes = roots;
        var n = word.Length;
        for (int i = 0; i < n; i++)
        {
            var c = GetIndex(word[i]);
            if (nodes[c] is null)
            {
                nodes[c] = new TrieNode();
            }

            if (i == n - 1)
            {
                nodes[c].IsWordEnd = true;
            }
            nodes = nodes[c].Next;
        }
    }

    public bool Search(string word)
    {
        var nodes = roots;
        var n = word.Length;
        for (int i = 0; i < n; i++)
        {
            var c = GetIndex(word[i]);
            if (nodes[c] is null) return false;
            if (i == n - 1) return nodes[c].IsWordEnd;
            nodes = nodes[c].Next;
        }
        return false;
    }

    public bool StartsWith(string prefix)
    {
        var nodes = roots;
        var n = prefix.Length;
        for (int i = 0; i < n; i++)
        {
            var c = GetIndex(prefix[i]);
            if (nodes[c] is null) return false;
            nodes = nodes[c].Next;
        }
        return true;
    }

    private int GetIndex(char c)
    {
        return (int)(c - 'a');
    }
}

public class TrieNode
{
    public TrieNode[] Next { get; set; }
    public bool IsWordEnd { get; set; }
    public TrieNode()
    {
        Next = new TrieNode[26];
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.Insert(word);
 * bool param_2 = obj.Search(word);
 * bool param_3 = obj.StartsWith(prefix);
 */

