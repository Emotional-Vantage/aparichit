import unittest
import sys
import io
from unittest.mock import patch

try:
    import torch
    from unlearn_ai.cli import main
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class TestCLI(unittest.TestCase):

    @unittest.skipUnless(HAS_DEPS, "torch must be installed for CLI tests")
    def test_cli_help(self):
        with patch.object(sys, "argv", ["aparichit", "--help"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)

    @unittest.skipUnless(HAS_DEPS, "torch must be installed for CLI tests")
    def test_cli_no_args_shows_help(self):
        with patch.object(sys, "argv", ["aparichit"]):
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)

    @unittest.skipUnless(HAS_DEPS, "torch must be installed for CLI tests")
    def test_cli_subcommand_helps(self):
        for subcmd in ["benchmark", "evaluate", "audit", "unlearn"]:
            with patch.object(sys, "argv", ["aparichit", subcmd, "--help"]):
                with patch("sys.stdout", new_callable=io.StringIO):
                    with patch("sys.stderr", new_callable=io.StringIO):
                        with self.assertRaises(SystemExit) as cm:
                            main()
                        self.assertEqual(cm.exception.code, 0)

    @unittest.skipUnless(HAS_DEPS, "torch must be installed for CLI tests")
    def test_cli_evaluate_arg_parser(self):
        with patch.object(sys, "argv", ["aparichit", "evaluate", "--checkpoint", "checkpoints/m_unlearned_gd", "--audit", "--max-generate", "2"]):
            with patch("unlearn_ai.cli.run_single_eval") as mock_eval:
                main()
                mock_eval.assert_called_once()
                args = mock_eval.call_args[0][0]
                self.assertEqual(args.checkpoint, "checkpoints/m_unlearned_gd")
                self.assertTrue(args.audit)
                self.assertEqual(args.max_generate, 2)


if __name__ == "__main__":
    unittest.main()
