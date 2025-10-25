# -Spell_Checker-
Intelligent Spell Checker using NLP, Trie, and Bayesian Probability:
This project implements an intelligent spell checker that uses a combination of linguistic probability, phonetic similarity, and edit distance to suggest the most likely corrections for misspelled words.

It is inspired by how real-world systems like Google’s spell correction ("Did you mean…") work — combining data structures, probability theory, and phonetics.
Trie-based Dictionary

Efficiently stores and searches through thousands of valid English words.

--> Weighted Levenshtein Distance

Computes the minimal number of edits (insertions, deletions, replacements) required to transform one word into another, with adaptive weighting based on word length.

--> Soundex Phonetic Algorithm

Detects words that “sound similar” (e.g., fone → phone).

--> Bayesian Probability Ranking

Scores candidate corrections using:

P(word∣misspelling)=P(word)×e−distance×P(context)
P(word∣misspelling)=P(word)×exp(−distance)×P(context)

which balances frequency, edit similarity, and contextual fit.

--> Context-Aware Suggestions

Uses bigram probabilities from the Reuters corpus to understand the likelihood of a word following another (e.g., bank near money vs bank near river).
