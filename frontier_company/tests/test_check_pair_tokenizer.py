import unittest

from frontier_company.check_pair_tokenizer import check_compatible


class FakeTokenizer:
    def __init__(self, vocab=None, eos=2, template_ids=None):
        self._vocab = vocab or {"a": 0, "b": 1, "<eos>": 2}
        self.eos_token_id = eos
        self.bos_token_id = None
        self.pad_token_id = 2
        self._template_ids = template_ids or [0, 1]

    def get_vocab(self):
        return self._vocab

    def apply_chat_template(self, *args, **kwargs):
        return self._template_ids


class PairTokenizerTest(unittest.TestCase):
    def test_accepts_identical_token_ids_and_thinking_template(self):
        check_compatible(FakeTokenizer(), FakeTokenizer())

    def test_rejects_different_token_ids_even_when_vocab_size_matches(self):
        with self.assertRaisesRegex(ValueError, "token IDs"):
            check_compatible(
                FakeTokenizer(),
                FakeTokenizer(vocab={"a": 1, "b": 0, "<eos>": 2}),
            )

    def test_rejects_different_thinking_prompt_template(self):
        with self.assertRaisesRegex(ValueError, "thinking chat template"):
            check_compatible(
                FakeTokenizer(),
                FakeTokenizer(template_ids=[0, 0, 1]),
            )

    def test_rejects_different_special_token_ids(self):
        with self.assertRaisesRegex(ValueError, "special token IDs"):
            check_compatible(FakeTokenizer(), FakeTokenizer(eos=1))


if __name__ == "__main__":
    unittest.main()
