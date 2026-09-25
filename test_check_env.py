import unittest

from check_env import summary


class CheckEnvTest(unittest.TestCase):
    def test_summary_mentions_python(self) -> None:
        self.assertTrue(summary().startswith("Python "))


if __name__ == "__main__":
    unittest.main()
