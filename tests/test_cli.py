import unittest
import sys
import io
from unittest.mock import patch
from unlearn_ai.cli import main


class TestCLI(unittest.TestCase):

    def test_cli_help(self):
        with patch.object(sys, "argv", ["aparichit", "--help"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)

    def test_cli_no_args_shows_help(self):
        with patch.object(sys, "argv", ["aparichit"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)

    def test_cli_subcommand_helps(self):
        for subcmd in ["benchmark", "evaluate", "audit", "unlearn"]:
            with patch.object(sys, "argv", ["aparichit", subcmd, "--help"]):
                with patch("sys.stdout", new_callable=io.StringIO):
                    with patch("sys.stderr", new_callable=io.StringIO):
                        with self.assertRaises(SystemExit) as cm:
                            main()
                        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
