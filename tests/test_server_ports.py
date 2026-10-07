"""Port collision handling without loading models or launching public tunnels."""

import errno
import unittest
from unittest.mock import Mock

from network_settings import launch_with_port_fallback


class PortFallbackTests(unittest.TestCase):
    def test_free_preferred_port(self):
        demo = Mock()
        demo.launch.return_value = ("app", "http://127.0.0.1:7861/", None)
        result = launch_with_port_fallback(demo, preferred_port=7861, share=False)
        self.assertEqual(result, demo.launch.return_value)
        demo.launch.assert_called_once_with(server_port=7861, share=False)

    def test_occupied_default_and_explicit_ports_try_next(self):
        for host in ("127.0.0.1", "0.0.0.0"):
            for share in (False, True):
                with self.subTest(host=host, share=share):
                    demo = Mock()
                    demo.launch.side_effect = [
                        OSError("Cannot find empty port in range: 7861-7861."),
                        ("app", "http://127.0.0.1:7862/", None),
                    ]
                    launch_with_port_fallback(
                        demo, preferred_port=7861, server_name=host, share=share,
                    )
                    self.assertEqual(
                        [call.kwargs["server_port"] for call in demo.launch.call_args_list],
                        [7861, 7862],
                    )
                    self.assertEqual(demo.launch.call_args.kwargs["server_name"], host)
                    self.assertEqual(demo.launch.call_args.kwargs["share"], share)

    def test_actual_address_in_use_retries(self):
        demo = Mock()
        demo.launch.side_effect = [OSError(errno.EADDRINUSE, "busy"), "started"]
        self.assertEqual(launch_with_port_fallback(demo, preferred_port=7860), "started")

    def test_unrelated_errors_are_not_hidden(self):
        demo = Mock()
        demo.launch.side_effect = OSError("Model file missing")
        with self.assertRaisesRegex(OSError, "Model file missing"):
            launch_with_port_fallback(demo, preferred_port=7860)
        self.assertEqual(demo.launch.call_count, 1)

    def test_exhaustion_is_bounded(self):
        demo = Mock()
        demo.launch.side_effect = OSError("Cannot find empty port in range: busy")
        with self.assertRaisesRegex(OSError, "No available port from 7861 to 7862"):
            launch_with_port_fallback(demo, preferred_port=7861, attempts=2)
        self.assertEqual(demo.launch.call_count, 2)

    def test_upper_port_limit(self):
        demo = Mock()
        demo.launch.side_effect = OSError("Cannot find empty port in range: busy")
        with self.assertRaises(OSError):
            launch_with_port_fallback(demo, preferred_port=65535, attempts=100)
        self.assertEqual(demo.launch.call_count, 1)

    def test_invalid_configuration_does_not_launch(self):
        for port, attempts in ((0, 100), (65536, 100), (7860, 0)):
            demo = Mock()
            with self.assertRaises(ValueError):
                launch_with_port_fallback(demo, preferred_port=port, attempts=attempts)
            demo.launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

