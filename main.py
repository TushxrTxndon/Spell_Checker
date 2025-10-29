"""
Advanced Finite Automata-Based Spell Checker with Probabilistic Correction
Combines DFA structure with Bayesian probability and phonetic matching
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import nltk
        
# Try to import optional libraries
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None
    print("⚠️  python-Levenshtein not found. Using fallback implementation.")

try:
    import phonetics
    PHONETICS_AVAILABLE = True
except ImportError:
    PHONETICS_AVAILABLE = False
    print("⚠️  phonetics library not found. Install with: pip install phonetics")

# Download NLTK data
nltk.download('words', quiet=True)
nltk.download('reuters', quiet=True)
from nltk.corpus import words, reuters
from nltk import bigrams


# ===================== TRIE/DFA IMPLEMENTATION =====================
class TrieNode:
    """Node representing a state in the DFA"""
    __slots__ = ['children', 'is_end_of_word', 'frequency']
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.frequency: int = 0


class Trie:
    """Trie data structure (DFA) for dictionary representation"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str, frequency: int = 1):
        """Insert word into the DFA with frequency"""
        if not word:
            return
        
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        node.is_end_of_word = True
        node.frequency += frequency
    
    def search(self, word: str) -> bool:
        """Search for word in dictionary (DFA traversal)"""
        if not word:
            return False
        
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def get_frequency(self, word: str) -> int:
        """Get word frequency from trie"""
        if not word:
            return 0
        
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.frequency if node.is_end_of_word else 0
    
    def get_all_words(self, node=None, prefix="") -> List[str]:
        """Get all words in dictionary"""
        if node is None:
            node = self.root
        
        words_list = []
        if node.is_end_of_word:
            words_list.append(prefix)
        
        for char, child in node.children.items():
            words_list.extend(self.get_all_words(child, prefix + char))
        
        return words_list


# ===================== FREQUENCY MAPS =====================
def build_frequency_maps(corpus_words):
    """Build unigram and bigram probability maps"""
    unigram = Counter(corpus_words)
    bigram_counter = Counter(bigrams(corpus_words))
    
    total_unigrams = sum(unigram.values())
    vocab_size = len(unigram)
    
    # Unigram probabilities with Laplace smoothing
    unigram_prob = {
        w: (c + 1) / (total_unigrams + vocab_size) 
        for w, c in unigram.items()
    }
    
    # Bigram probabilities
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 1e-10))
    for (w1, w2), c in bigram_counter.items():
        if unigram[w1] > 0:
            bigram_prob[w1][w2] = (c + 1) / (unigram[w1] + vocab_size)
    
    return unigram_prob, bigram_prob


# ===================== EDIT DISTANCE =====================
def compute_levenshtein_distance(w1: str, w2: str) -> int:
    """
    Compute Levenshtein edit distance.
    Falls back to DP implementation if library not available.
    """
    if levenshtein_distance:
        return levenshtein_distance(w1, w2)
    
    # Fallback DP implementation
    if not w1:
        return len(w2)
    if not w2:
        return len(w1)
    
    dp = [[0] * (len(w2) + 1) for _ in range(len(w1) + 1)]
    
    for i in range(len(w1) + 1):
        dp[i][0] = i
    for j in range(len(w2) + 1):
        dp[0][j] = j
    
    for i in range(1, len(w1) + 1):
        for j in range(1, len(w2) + 1):
            if w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )
    
    return dp[len(w1)][len(w2)]


def weighted_edit_distance(w1: str, w2: str) -> float:
    """Calculate weighted edit distance with length normalization"""
    base_distance = compute_levenshtein_distance(w1, w2)
    
    # Length-based weighting
    avg_len = (len(w1) + len(w2)) / 2
    if avg_len == 0:
        return float('inf')
    
    # Penalty for length difference
    len_diff = abs(len(w1) - len(w2))
    len_penalty = len_diff * 0.1
    
    return base_distance + len_penalty


# ===================== PHONETIC ALGORITHMS =====================
def soundex(word: str) -> str:
    """
    Generate Soundex code for phonetic matching.
    Groups words that sound similar.
    """
    if not word or not word.isalpha():
        return "0000"
    
    word = word.upper()
    first_letter = word[0]
    
    # Soundex character mapping
    soundex_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    code = []
    prev_digit = soundex_map.get(first_letter, '0')
    
    for char in word[1:]:
        digit = soundex_map.get(char, '0')
        if digit != '0' and digit != prev_digit:
            code.append(digit)
        if char not in 'AEIOUHWY':
            prev_digit = digit
    
    code_str = ''.join(code)
    return (first_letter + code_str + '000')[:4]


def get_metaphone(word: str) -> str:
    """Get Metaphone code if library available"""
    if PHONETICS_AVAILABLE:
        try:
            return phonetics.metaphone(word)
        except:
            return soundex(word)
    return soundex(word)


# ===================== BAYESIAN SCORING =====================
def bayesian_score(word: str, misspelled: str, freq_map: dict, 
                   dist: float, bigram_prob: dict = None, 
                   prev_word: str = None) -> float:
    """
    Calculate Bayesian probability score for correction candidate.
    
    Score = P(word) × P(misspelled|word) × P(word|prev_word)
    """
    # Prior: Word frequency
    P_w = freq_map.get(word, 1e-10)
    
    # Likelihood: Exponential decay with distance
    P_x_given_w = math.exp(-dist * 1.5)
    
    # Context probability
    if prev_word and bigram_prob:
        P_context = bigram_prob.get(prev_word, {}).get(word, 1e-10)
    else:
        P_context = 1.0
    
    return P_w * P_x_given_w * P_context


# ===================== SPELL CHECKER CLASS =====================
class AdvancedSpellChecker:
    """
    Advanced spell checker combining DFA with probabilistic correction
    """
    
    def __init__(self, wordlist=None, load_corpus=True):
        print("🔧 Initializing Advanced Spell Checker...")
        
        # Build dictionary
        if wordlist is None:
            print("📚 Loading dictionary...")
            raw_words = words.words()
            wordlist = [
                w.lower() for w in raw_words 
                if isinstance(w, str) and w.isalpha() and len(w) > 1
            ]
        
        # Initialize Trie (DFA)
        self.trie = Trie()
        self.dictionary = set(wordlist)
        
        if load_corpus:
            # Build corpus statistics
            print("📊 Building frequency maps from corpus...")
            corpus_words = [w.lower() for w in reuters.words() if w.isalpha()]
            self.unigram_prob, self.bigram_prob = build_frequency_maps(corpus_words)
            
            # Insert words with frequencies
            for w in wordlist:
                freq = int(self.unigram_prob.get(w, 1e-6) * 1000000)
                self.trie.insert(w, max(freq, 1))
        else:
            self.unigram_prob = {w: 1.0 / len(wordlist) for w in wordlist}
            self.bigram_prob = defaultdict(lambda: defaultdict(lambda: 1e-10))
            for w in wordlist:
                self.trie.insert(w)
        
        # Build phonetic indices
        print("🎵 Building phonetic indices...")
        self.soundex_map = defaultdict(set)
        self.metaphone_map = defaultdict(set)
        
        for w in self.dictionary:
            self.soundex_map[soundex(w)].add(w)
            if PHONETICS_AVAILABLE:
                self.metaphone_map[get_metaphone(w)].add(w)
        
        print(f"✅ Spell checker initialized with {len(self.dictionary)} words\n")
    
    def search(self, word: str) -> bool:
        """Check if word is in dictionary"""
        return self.trie.search(word.lower())
    
    def insert(self, word: str):
        """Add word to dictionary"""
        word = word.lower().strip()
        if word and word.isalpha():
            self.trie.insert(word)
            self.dictionary.add(word)
            self.soundex_map[soundex(word)].add(word)
            if PHONETICS_AVAILABLE:
                self.metaphone_map[get_metaphone(word)].add(word)
    
    def get_candidates(self, word: str, max_dist: int = 2) -> List[Tuple[str, float]]:
        """
        Find correction candidates using multiple strategies:
        1. Phonetic matching (Soundex + Metaphone)
        2. Length-based filtering
        3. Edit distance computation
        """
        word_lower = word.lower()
        candidates = set()
        
        # Strategy 1: Phonetic candidates
        sx = soundex(word_lower)
        candidates.update(self.soundex_map.get(sx, set()))
        
        if PHONETICS_AVAILABLE:
            mp = get_metaphone(word_lower)
            candidates.update(self.metaphone_map.get(mp, set()))
        
        # Strategy 2: Length-based filtering (fallback)
        if not candidates or len(candidates) < 5:
            length_candidates = {
                w for w in self.dictionary 
                if abs(len(w) - len(word_lower)) <= max_dist
            }
            candidates.update(length_candidates)
        
        # Strategy 3: First letter matching
        if word_lower:
            first_letter_candidates = {
                w for w in self.dictionary 
                if w[0] == word_lower[0] and abs(len(w) - len(word_lower)) <= max_dist
            }
            candidates.update(first_letter_candidates)
        
        # Compute edit distances
        valid_candidates = []
        for candidate in candidates:
            dist = weighted_edit_distance(word_lower, candidate)
            if dist <= max_dist:
                valid_candidates.append((candidate, dist))
        
        return valid_candidates
    
    def correct(self, misspelled: str, prev_word: str = None, 
                topn: int = 3) -> List[str]:
        """
        Generate top-N spelling corrections with Bayesian ranking.
        
        Args:
            misspelled: The word to correct
            prev_word: Previous word for context-aware correction
            topn: Number of suggestions to return
            
        Returns:
            List of correction suggestions ranked by probability
        """
        if not misspelled or not misspelled.strip():
            return []
        
        misspelled = misspelled.strip().lower()
        
        # Check if word is already correct
        if self.trie.search(misspelled):
            return [misspelled]
        
        # Get candidates
        candidates = self.get_candidates(misspelled, max_dist=2)
        
        if not candidates:
            return []
        
        # Score candidates using Bayesian inference
        scored = []
        for word, dist in candidates:
            score = bayesian_score(
                word, misspelled, self.unigram_prob, dist,
                bigram_prob=self.bigram_prob, prev_word=prev_word
            )
            scored.append((score, dist, word))
        
        # Sort by score (descending), then distance (ascending)
        scored.sort(key=lambda x: (-x[0], x[1]))
        
        return [word for _, _, word in scored[:topn]]
    
    def correct_with_details(self, misspelled: str, prev_word: str = None, 
                            topn: int = 5) -> List[Dict]:
        """Return corrections with detailed scoring information"""
        if not misspelled or not misspelled.strip():
            return []
        
        misspelled = misspelled.strip().lower()
        
        if self.trie.search(misspelled):
            return [{'word': misspelled, 'edit_distance': 0, 'score': 1.0, 'status': 'correct'}]
        
        candidates = self.get_candidates(misspelled, max_dist=2)
        
        if not candidates:
            return []
        
        scored = []
        for word, dist in candidates:
            score = bayesian_score(
                word, misspelled, self.unigram_prob, dist,
                bigram_prob=self.bigram_prob, prev_word=prev_word
            )
            scored.append({
                'word': word,
                'edit_distance': dist,
                'score': score,
                'frequency': self.unigram_prob.get(word, 1e-10),
                'soundex': soundex(word)
            })
        
        scored.sort(key=lambda x: (-x['score'], x['edit_distance']))
        return scored[:topn]
    
    def get_dictionary_size(self) -> int:
        """Get number of words in dictionary"""
        return self.trie.word_count


# ===================== MENU-DRIVEN APPLICATION =====================
class SpellCheckerApp:
    """Interactive spell checker application"""
    
    def __init__(self):
        self.checker = AdvancedSpellChecker()
    
    def _display_menu(self):
        """Display main menu"""
        print("\n╔═══════════════════════════════════════════════════════╗")
        print("║   ADVANCED DFA-BASED SPELL CHECKER                   ║")
        print("║   with Bayesian Probability & Phonetic Matching      ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("\n1. Check Spelling (Single Word)")
        print("2. Check Spelling (Full Text)")
        print("3. Context-Aware Correction")
        print("4. Add Word to Dictionary")
        print("5. Search Word in Dictionary")
        print("6. View Dictionary Stats")
        print("7. Get Detailed Suggestions")
        print("8. Run Test Cases")
        print("9. Exit")
        print("\nEnter your choice: ", end="")
    
    def run(self):
        """Run the application"""
        print("\n🔤 Welcome to Advanced Spell Checker!")
        print(f"Dictionary loaded with {self.checker.get_dictionary_size()} words")
        
        while True:
            self._display_menu()
            
            try:
                choice = input().strip()
                
                if choice == '1':
                    self.check_single_word()
                elif choice == '2':
                    self.check_full_text()
                elif choice == '3':
                    self.context_aware_correction()
                elif choice == '4':
                    self.add_word()
                elif choice == '5':
                    self.search_word()
                elif choice == '6':
                    self.show_stats()
                elif choice == '7':
                    self.detailed_suggestions()
                elif choice == '8':
                    self.run_tests()
                elif choice == '9':
                    print("\n👋 Thank you for using Spell Checker!")
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using Spell Checker!")
                break
            except Exception as e:
                print(f"\n⚠️  An error occurred: {e}")
    
    def check_single_word(self):
        """Check spelling of a single word"""
        print("\n📝 Enter word to check: ", end="")
        word = input().strip()
        
        if not word:
            print("⚠️  No word entered.")
            return
        
        if self.checker.search(word):
            print(f"✅ \"{word}\" is spelled correctly!")
        else:
            print(f"❌ \"{word}\" is misspelled.")
            suggestions = self.checker.correct(word, topn=5)
            if suggestions:
                print(f"💡 Suggestions: {', '.join(suggestions)}")
            else:
                print("⚠️  No suggestions found.")
    
    def check_full_text(self):
        """Check spelling of full text"""
        print("\n📝 Enter text to check: ", end="")
        text = input().strip()
        
        if not text:
            print("⚠️  No text entered.")
            return
        
        words = re.findall(r'[a-zA-Z]+', text)
        correct = 0
        incorrect = 0
        
        print("\n" + "=" * 70)
        print("SPELL CHECK RESULTS")
        print("=" * 70 + "\n")
        
        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else None
            
            if self.checker.search(word):
                print(f"✓ \"{word}\" - CORRECT")
                correct += 1
            else:
                print(f"✗ \"{word}\" - INCORRECT")
                suggestions = self.checker.correct(word, prev_word=prev_word, topn=3)
                if suggestions:
                    print(f"  💡 Suggestions: {', '.join(suggestions)}")
                incorrect += 1
            print()
        
        print("=" * 70)
        print(f"📊 Statistics:")
        print(f"   Total words: {len(words)}")
        print(f"   Correct: {correct}")
        print(f"   Incorrect: {incorrect}")
        if words:
            print(f"   Accuracy: {(correct * 100.0 / len(words)):.2f}%")
        print("=" * 70)
    
    def context_aware_correction(self):
        """Demonstrate context-aware correction"""
        print("\n📝 Enter previous word (for context): ", end="")
        prev_word = input().strip()
        print("📝 Enter word to correct: ", end="")
        word = input().strip()
        
        if not word:
            print("⚠️  No word entered.")
            return
        
        suggestions = self.checker.correct(word, prev_word=prev_word or None, topn=5)
        
        print(f"\n🔍 Context: \"{prev_word or 'None'}\"")
        print(f"🔍 Word: \"{word}\"")
        print(f"✅ Suggestions: {', '.join(suggestions) if suggestions else 'None'}")
    
    def add_word(self):
        """Add word to dictionary"""
        print("\n➕ Enter word to add: ", end="")
        word = input().strip()
        
        if not word:
            print("⚠️  No word entered.")
            return
        
        if self.checker.search(word):
            print(f"ℹ️  \"{word}\" already exists in dictionary.")
        else:
            self.checker.insert(word)
            print(f"✅ \"{word}\" added successfully!")
            print(f"📚 Dictionary size: {self.checker.get_dictionary_size()} words")
    
    def search_word(self):
        """Search for word in dictionary"""
        print("\n🔍 Enter word to search: ", end="")
        word = input().strip()
        
        if not word:
            print("⚠️  No word entered.")
            return
        
        if self.checker.search(word):
            print(f"✅ \"{word}\" is in the dictionary!")
        else:
            print(f"❌ \"{word}\" is NOT in the dictionary.")
            suggestions = self.checker.correct(word, topn=3)
            if suggestions:
                print(f"💡 Did you mean: {', '.join(suggestions)}?")
    
    def show_stats(self):
        """Show dictionary statistics"""
        print("\n📊 DICTIONARY STATISTICS")
        print("-" * 50)
        print(f"Total words: {self.checker.get_dictionary_size()}")
        print(f"Structure: Trie-based DFA")
        print(f"Features:")
        print(f"  • Bayesian probability scoring")
        print(f"  • Context-aware correction (bigrams)")
        print(f"  • Phonetic matching (Soundex{' + Metaphone' if PHONETICS_AVAILABLE else ''})")
        print(f"  • Weighted edit distance")
        print("-" * 50)
    
    def detailed_suggestions(self):
        """Get detailed suggestions with scoring"""
        print("\n💡 Enter word to analyze: ", end="")
        word = input().strip()
        
        if not word:
            print("⚠️  No word entered.")
            return
        
        details = self.checker.correct_with_details(word, topn=5)
        
        print(f"\n🔍 Detailed Analysis for \"{word}\":")
        print("=" * 70)
        
        if not details:
            print("⚠️  No suggestions found.")
        else:
            for i, item in enumerate(details, 1):
                print(f"\n{i}. {item['word']}")
                print(f"   Edit Distance: {item['edit_distance']}")
                print(f"   Bayesian Score: {item['score']:.8f}")
                print(f"   Frequency: {item['frequency']:.8f}")
                print(f"   Soundex: {item['soundex']}")
        
        print("\n" + "=" * 70)
    
    def run_tests(self):
        """Run predefined test cases"""
        tests = [
            ("rate", "animal"),
            ("bananna", None),
            ("fone", None),
            ("hosre", "ride"),
            ("bank", "river"),
            ("bank", "money"),
            ("speling", None),
            ("recieve", None),
            ("definately", None),
        ]
        
        print("\n" + "=" * 70)
        print("🔎 RUNNING TEST CASES")
        print("=" * 70 + "\n")
        
        for word, context in tests:
            suggestions = self.checker.correct(word, prev_word=context, topn=3)
            print(f"❌ Misspelled: {word}")
            print(f"🧩 Context: {context or 'None'}")
            print(f"✅ Suggestions: {', '.join(suggestions) if suggestions else 'None'}\n")
        
        print("=" * 70)


# ===================== MAIN =====================
def main():
    """Main entry point"""
    app = SpellCheckerApp()
    app.run()


if __name__ == "__main__":
    main()
