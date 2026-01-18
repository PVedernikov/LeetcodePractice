namespace LeetcodePreapare;

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