"""Levenshtein Distance is the minumum number of operations(insertions,deletions,replacements)
on a single character of a string s to convert it to string t
For example:s="Aditya", t="Advik"
We notice that Ad is common, to convert s->t perform the following operations:
1. Replace i with v
2. Replace t with i
3. Replace y with k
3. Delete a
So the Levenshtein Distance between s and t is 3"""
"""We will implement the spell checker using 4 parameters:
1. A Trie to represent the dictionary of valid words.
2. Frequency map to Store the probabilies of words
3. Weighted Levenshtein Distance to find the best candidate corrections
4. Bayesian Probability to rank the candidate corrections
5. Soundex Algorithm to find phonetically siilar words like "phone" and "fone" """
import math
from collections import Counter, defaultdict
from Levenshtein import distance as levenshtein_distance
import nltk
nltk.download('words', quiet=True)
nltk.download('reuters', quiet=True)
from nltk.corpus import words, reuters
from nltk import bigrams
class Node:
    """This is a node of the trie data structure which would represent
      a state in the DFA"""
    def __init__(self):
        self.children = dict
        self.is_end_of_word = False
class Trie:
    def __init__ (self):
        self.root=Node()
    def insert(self,word:str):
        curr_node=self.root
        for ch in word:
            if ch not in curr_node.children:
                curr_node.children[ch]=Node()
            curr_node=curr_node.children[ch]
        curr_node.is_end_of_word=True
    def search(self,word:str):
        curr_Node=self.root
        for ch in word:
            if ch not in curr_Node.children:
                return False
            curr_Node=curr_Node.children[ch] 
        return curr_Node.is_end_of_word 
def build_frequency_maps(corpus_words):
    unigram= Counter(corpus_words)
    bigram =Counter(bigrams(corpus_words))  
    total_unigrams = sum(unigram.values())
    unigram_prob = {w:c/total_unigrams for w,c in unigram.items()}
    bigram_prob = defaultdict(lambda: defaultdict(1e-9))
    for (w1,w2),c in bigram.items():
        bigram_prob[w1][w2] = c /unigram
    return unigram_prob, bigram_prob
def weighted_edit_distance(w1,w2):
    base= levenshtein_distance(w1,w2)
    weight = 1.0 - (0.1 * min(len(w1), len(w2)) / 10)
    return base * weight 
def soundex(word):
    word=word.lower()
    first_letter=word[0].upper()
    replacements ={'bfpv': '1', 'cgjkqsxz': '2', 'dt': '3', 'l': '4',
        'mn': '5', 'r': '6'}
    code = ''
    for ch in word[1:]:
        for key, val in replacements.items():
            if ch in key:
                if not code or val != code[-1]:
                    code += val
                break
    code =code.replace('0','')
    return(first_letter + code + '000')[:4]
def bayesian_score(word,misspelled,freq_map,dist,bigram_prob=None,prev_word=None):
    P_w=freq_map.get(word,1e-9)
    #Find probability of correction given the misspelled word P(x|w)
    #Use exp(-dist) since it decays much faster with each additional edit
    P_x_given_w = math.exp(-dist)
    #Find context probability P(word|pre_word)
    if prev_word and bigram_prob:
        P_context = bigram_prob.get((prev_word,word),1e-9)
    else :
        P_context = 1.0
    """Return the final Score which describes:
    1.How common the word is in general usage
    2.How similar the candidate correction is to the misspelled word
    3.How well the candidate correction fits in the context of the sentence"""
    return P_w * P_x_given_w * P_context
class SpellChecker:
    def __init__(self):
        self.dictionary = [w.lower() for w in words.words() if w.isalpha()]
        self.trie = Trie()
        for w in self .dictionary:
            self.trie.insert(w)
        corpus = [w.lower() for w in reuters.words() if w.isalpha()]
        self.unigram_prob, self.bigram_prob = build_frequency_maps(corpus)
        self.soundex_map = defaultdict(set)
        for w in self.dictionary:
            self.soundex_map[soundex(w)].add(w)
    def get_candidates(self, word, max_dist=2):
        candidates = []
        code = soundex(word)
        nearby_words = self.soundex_map.get(code, set()) | set(self.dictionary)
        for dw in nearby_words:
            if abs(len(dw) - len(word)) <= max_dist:
                dist = weighted_edit_distance(word, dw)
                if dist <= max_dist:
                    candidates.append((dw, dist))
        return candidates

    def correct(self, misspelled, prev_word=None, topn=3):
        if self.trie.search(misspelled):
            return [misspelled]
        
        candidates = self.get_candidates(misspelled, max_dist=2)
        if not candidates:
            return []
        
        scored = []
        for w, dist in candidates:
            score = bayesian_score(
                w, misspelled,
                self.unigram_prob,
                dist,
                bigram_prob=self.bigram_prob,
                prev_word=prev_word
            )
            scored.append((score, w))
        
        scored.sort(reverse=True)
        return [w for _, w in scored[:topn]]
if __name__ == "__main__":
    checker = SpellChecker()
    
    tests = [
        ("aplpe", None),
        ("bananna", None),
        ("fone", None),
        ("hosre", "ride"),     
        ("bank", "river"),       
        ("bank", "money"),     
    ]
    print("Spell Checker Suggestions:\n")
    
    for word, context in tests:
        suggestions = checker.correct(word, prev_word=context, topn=3)
        print(f"❌ Misspelled: {word}")
        print(f"🧩 Context: {context}")
        print(f"✅ Suggestions: {suggestions}\n")