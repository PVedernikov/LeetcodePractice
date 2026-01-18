namespace LeetcodePreapare;

// Blind 75 
// https://leetcode.com/problem-list/r3q9lspc/
// #53
// 211. Design Add and Search Words Data Structure
// Design a data structure that supports adding new words and finding if a string matches any previously added string.
// Implement the WordDictionary class:
// - WordDictionary() Initializes the object.
// - void addWord(word) Adds word to the data structure, it can be matched later.
// - bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise.
//   word may contain dots '.' where dots can be matched with any letter.

public class WordDictionary
{
    private TrieNodeWD[] roots;

    public WordDictionary()
    {
        roots = new TrieNodeWD[26];
    }

    public void AddWord(string word)
    {
        var nodes = roots;
        var n = word.Length;
        for (int i = 0; i < n; i++)
        {
            var c = (int)(word[i] - 'a');
            if (nodes[c] is null)
            {
                nodes[c] = new TrieNodeWD();
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
        return Search(word, 0, roots);
    }

    private bool Search(string word, int i, TrieNodeWD[] nodes)
    {
        if (word[i] != '.')
        {
            var c = (int)(word[i] - 'a');
            if (nodes[c] is null) return false;
            if (i == word.Length - 1) return nodes[c].IsWordEnd;
            return Search(word, i + 1, nodes[c].Next);
        }

        if (i == word.Length - 1) return nodes.Any(x => x is not null && x.IsWordEnd);

        for (int j = 0; j < 26; j++)
        {
            if (nodes[j] is not null && Search(word, i + 1, nodes[j].Next))
            {
                return true;
            }
        }

        return false;
    }
}

public class TrieNodeWD
{
    public TrieNodeWD[] Next { get; set; }
    public bool IsWordEnd { get; set; }
    public TrieNodeWD()
    {
        Next = new TrieNodeWD[26];
    }
}
/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.AddWord(word);
 * bool param_2 = obj.Search(word);
 */
