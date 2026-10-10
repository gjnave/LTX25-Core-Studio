"""Gated-file authorization checks without downloading weights or saving tokens."""
from pathlib import Path
import hashlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

import app
import matte_tools as tools


class AdapterAccessTests(unittest.TestCase):
    def test_download_verifies_checksum_on_python_310(self):
        folder=Path(tempfile.mkdtemp(prefix='spokesman-download-check-'))
        weights=folder/'adapter.safetensors'
        payload=b'checksum-test'
        weights.write_bytes(payload)
        with patch.object(tools,'adapter_ready',return_value=False),patch.object(tools,'check_adapter_access',return_value={'allowed':True}),patch.object(tools,'ALPHA_SIZE',len(payload)),patch.object(tools,'ALPHA_SHA256',hashlib.sha256(payload).hexdigest()),patch('huggingface_hub.hf_hub_download',return_value=str(weights)):
            self.assertIn('verified',tools.download_adapter(folder/'models','test-token'))
        self.assertFalse(tools.DOWNLOAD_LOCK.locked())
    def test_no_login_explains_same_account_setup_without_network(self):
        with patch('huggingface_hub.get_token',return_value=None),patch('huggingface_hub.get_hf_file_metadata') as request:
            result=tools.check_adapter_access('')
        self.assertFalse(result['allowed'])
        self.assertEqual(result['status'],'no_login')
        request.assert_not_called()

    def test_saved_login_checks_actual_pinned_weights_without_download(self):
        metadata=SimpleNamespace(size=tools.ALPHA_SIZE,etag=tools.ALPHA_SHA256)
        with patch('huggingface_hub.get_token',return_value='saved-test-token'),patch('huggingface_hub.get_hf_file_metadata',return_value=metadata) as request,patch('huggingface_hub.hf_hub_download') as download:
            result=tools.check_adapter_access()
        self.assertTrue(result['allowed'])
        self.assertEqual(request.call_args.kwargs['token'],'saved-test-token')
        self.assertIn(tools.ALPHA_REVISION,request.call_args.args[0])
        self.assertTrue(request.call_args.args[0].endswith(tools.ALPHA_FILE))
        download.assert_not_called()

    def test_denied_invalid_and_connection_failures_are_distinct_and_sanitized(self):
        response=httpx.Response(401,request=httpx.Request('HEAD','https://huggingface.co/'))
        gated_response=httpx.Response(403,request=httpx.Request('HEAD','https://huggingface.co/'))
        errors=[(GatedRepoError('secret-test-token',response=gated_response),'access_denied'),
                (HfHubHTTPError('secret-test-token',response=response),'invalid_token'),
                (ConnectionError('secret-test-token'),'unavailable')]
        for error,status in errors:
            with self.subTest(status=status),patch('huggingface_hub.get_hf_file_metadata',side_effect=error):
                result=tools.check_adapter_access('secret-test-token')
                self.assertFalse(result['allowed'])
                self.assertEqual(result['status'],status)
                self.assertNotIn('secret-test-token',str(result))

    def test_access_updates_download_control_only_after_success(self):
        for allowed in [False,True]:
            with self.subTest(allowed=allowed),patch.object(app,'adapter_ready',return_value=False),patch.object(app,'check_adapter_access',return_value={'allowed':allowed,'message':'Checked'}):
                message,button=app.check_background_access('test-token')
                self.assertEqual(message,'Checked')
                self.assertEqual(button['interactive'],allowed)

    def test_download_rechecks_permission_before_writing_files(self):
        folder=Path(tempfile.mkdtemp(prefix='spokesman-access-check-'))/'models'
        with patch.object(tools,'check_adapter_access',return_value={'allowed':False,'message':'Approval pending'}),patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaisesRegex(ValueError,'Approval pending'):
                tools.download_adapter(folder,'test-token')
        download.assert_not_called()
        self.assertFalse(folder.exists())
        self.assertFalse(tools.DOWNLOAD_LOCK.locked())


if __name__=='__main__':
    unittest.main()
