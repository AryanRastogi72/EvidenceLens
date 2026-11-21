import unittest

from engine.controller import load_config, run_controller
from engine.conversation import SESSIONS, handle_turn
from engine.query_understanding import edit_distance, understand
from engine.search import SearchIndex
from evaluation.size_matched import bm25_top_k


class SpellingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex()
        cls.config = load_config()

    def concepts(self, text):
        return understand(text, self.index, self.config)

    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("vacine", "vaccine"), 1)
        self.assertEqual(edit_distance("same", "same"), 0)

    def test_a_missing_word_is_corrected_from_the_vocabulary(self):
        (concept,) = self.concepts("remdesivr")
        self.assertEqual(concept["stems"], ["remdesivir"])
        self.assertEqual(concept["corrected_from"], "remdesivr")
        self.assertEqual(concept["source"], "kgram")

    def test_a_rare_misspelling_that_is_in_the_index_is_corrected_too(self):
        (concept,) = self.concepts("vacine")
        self.assertEqual(concept["stems"], ["vaccin"])
        self.assertEqual(concept["label"], "vaccine")
        self.assertEqual(concept["corrected_from"], "vacine")

    def test_correct_words_are_left_alone(self):
        for text in ("covid", "vaccine", "masks"):
            self.assertTrue(all("corrected_from" not in c for c in self.concepts(text)), text)

    def test_a_word_with_no_close_candidate_is_skipped(self):
        self.assertEqual(self.concepts("qwertyzzzz"), [])

    def test_quoted_words_numbers_and_short_words_are_never_corrected(self):
        self.assertTrue(all("corrected_from" not in c for c in self.concepts('"vacine" masks')))
        self.assertTrue(all("corrected_from" not in c for c in self.concepts("covid 19")))
        self.assertTrue(all("corrected_from" not in c for c in self.concepts("xx")))

    def test_the_conversation_reports_a_spelling_fix_rule(self):
        result = handle_turn("diag_spelling", "vacine pregnant", self.index, self.config)
        self.assertIn({"rule": "SPELLING_FIX", "detail": "vacine -> vaccine"}, result["rules_fired"])
        SESSIONS.pop("diag_spelling", None)


class DisabledMovesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SearchIndex()
        cls.config = load_config()
        cls.text = "coronavirus asymptomatic transmission"

    def decisions(self, config):
        _, log, query = run_controller(self.text, self.index, config, allow_clarify=False)
        return [step["decision"] for step in log.steps], query

    def test_an_empty_list_changes_nothing(self):
        self.assertEqual(self.decisions(self.config), self.decisions(dict(self.config, disabled_moves=[])))

    def test_a_disabled_move_is_never_taken(self):
        self.assertIn("TIGHTEN_TITLE_ZONE", self.decisions(self.config)[0])
        self.assertNotIn("TIGHTEN_TITLE_ZONE", self.decisions(dict(self.config, disabled_moves=["TIGHTEN_TITLE_ZONE"]))[0])


class SizeMatchedTests(unittest.TestCase):
    def test_bm25_top_k_has_exactly_k_documents(self):
        index, config = SearchIndex(), load_config()
        self.assertEqual(len(bm25_top_k("covid vaccine pregnant", 25, index, config)), 25)
        self.assertEqual(bm25_top_k("covid vaccine pregnant", 0, index, config), set())


if __name__ == "__main__":
    unittest.main()
