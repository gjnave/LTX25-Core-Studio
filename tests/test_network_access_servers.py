"""Local owner access must remain independent of remote passwords."""
import unittest
from unittest.mock import Mock
from network_settings import launch_access_servers


class SeparateAccessTests(unittest.TestCase):
    def test_local_only_has_no_auth_or_public_tunnel(self):
        local = Mock()
        local.launch.return_value = ("app", "http://127.0.0.1:7860/", None)
        factory = Mock(return_value=local)
        result = launch_access_servers(factory, mode="local", preferred_port=7860, auth="remote-only")
        self.assertEqual(result, (local, None, "http://127.0.0.1:7860/", None))
        factory.assert_called_once()
        options = local.launch.call_args.kwargs
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertIsNone(options["auth"])
        self.assertFalse(options["share"])

    def test_public_password_only_applies_to_separate_remote_app(self):
        local, remote = Mock(), Mock()
        local.launch.return_value = ("app", "http://127.0.0.1:7860/", None)
        remote.launch.return_value = ("app", "http://127.0.0.1:8860/", "https://test.gradio.live")
        auth = Mock()
        result = launch_access_servers(
            Mock(side_effect=[local, remote]), mode="public", preferred_port=7860,
            auth=auth, inbrowser=True,
        )
        self.assertIsNone(local.launch.call_args.kwargs["auth"])
        self.assertFalse(local.launch.call_args.kwargs["share"])
        self.assertTrue(local.launch.call_args.kwargs["inbrowser"])
        self.assertIs(remote.launch.call_args.kwargs["auth"], auth)
        self.assertTrue(remote.launch.call_args.kwargs["share"])
        self.assertFalse(remote.launch.call_args.kwargs["inbrowser"])
        self.assertEqual(remote.launch.call_args.kwargs["server_port"], 8860)
        self.assertEqual(result[-1], "https://test.gradio.live")

    def test_lan_only_binds_authenticated_view_to_all_interfaces(self):
        local, remote = Mock(), Mock()
        local.launch.return_value = ("app", "http://127.0.0.1:7861/", None)
        remote.launch.return_value = ("app", "http://localhost:8861/", None)
        launch_access_servers(Mock(side_effect=[local, remote]), mode="lan",
                              preferred_port=7861, auth="remote-only")
        self.assertEqual(local.launch.call_args.kwargs["server_name"], "127.0.0.1")
        self.assertEqual(remote.launch.call_args.kwargs["server_name"], "0.0.0.0")
        self.assertEqual(remote.launch.call_args.kwargs["auth"], "remote-only")
        self.assertFalse(remote.launch.call_args.kwargs["share"])

    def test_failed_tunnel_keeps_local_app_open(self):
        local, remote = Mock(), Mock()
        local.launch.return_value = ("app", "http://127.0.0.1:7860/", None)
        remote.launch.return_value = ("app", "http://127.0.0.1:8860/", None)
        result = launch_access_servers(Mock(side_effect=[local, remote]), mode="public", preferred_port=7860)
        self.assertIsNone(result[1])
        remote.close.assert_called_once()
        local.close.assert_not_called()

    def test_failed_remote_start_keeps_local_app_open(self):
        local, remote = Mock(), Mock()
        local.launch.return_value = ("app", "http://127.0.0.1:7861/", None)
        remote.launch.side_effect = RuntimeError("network unavailable")
        result = launch_access_servers(Mock(side_effect=[local, remote]), mode="public", preferred_port=7861)
        self.assertEqual(result[2], "http://127.0.0.1:7861/")
        local.close.assert_not_called()

    def test_invalid_mode_does_not_launch(self):
        factory = Mock()
        with self.assertRaises(ValueError):
            launch_access_servers(factory, mode="invalid", preferred_port=7860)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

