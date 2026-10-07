import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import update_app as updater

class RepositoryUpdates(unittest.TestCase):
    def make_archive(self,cfg,extra=None):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:
            for name in cfg['required']:
                value=json.dumps(cfg) if name=='release_sources.json' else 'new source'
                z.writestr('app/'+name,value)
            for name,value in (extra or {}).items():
                z.writestr('app/'+name,value)
        return data.getvalue()

    def test_revision_without_version_bump_fallback_backup_and_private_preservation(self):
        root=Path(tempfile.mkdtemp(prefix='ggf-update-check-'))
        cfg={'mirrors':[{'name':'Codeberg','api':'bad','archive':'https://bad.invalid/{revision}'},
                        {'name':'GitHub','api':'good','archive':'https://good.invalid/{revision}'}],
             'required':['app.py','update_app.py','release_sources.json','VERSION','requirements.txt']}
        (root/'release_sources.json').write_text(json.dumps(cfg))
        (root/'app.py').write_text('old source')
        (root/'models').mkdir()
        (root/'models/keep').write_text('weights')
        (root/'network_settings.json').write_text('private')
        (root/updater.MARKER).write_text(json.dumps({'revision':'a'*40}))
        payload=self.make_archive(cfg,{'vendor/comfy_core/comfy/ldm/models/code.py':'code'})
        def latest(source):
            if source['name']=='Codeberg': raise OSError('unavailable')
            return 'b'*40
        with patch.object(updater,'latest',side_effect=latest),patch.object(updater,'urlopen',return_value=io.BytesIO(payload)):
            self.assertIn('Update available',updater.check_update(root))
            updater.update(root)
        self.assertEqual((root/'app.py').read_text(),'new source')
        self.assertEqual((root/'models/keep').read_text(),'weights')
        self.assertEqual((root/'network_settings.json').read_text(),'private')
        self.assertTrue(any(p.read_text()=='old source' for p in (root.parent/'app-backups').glob('*/app.py')))
        with patch.object(updater,'latest',side_effect=latest):
            self.assertIn('Up to date',updater.check_update(root))

    def test_invalid_archive_cannot_write_protected_paths(self):
        root=Path(tempfile.mkdtemp(prefix='ggf-invalid-update-'))
        cfg={'required':['app.py','update_app.py','release_sources.json']}
        for bad in ['../outside.py','network_settings.json','models/weights','C:/outside.py']:
            archive=root/(str(abs(hash(bad)))+'.zip')
            archive.write_bytes(self.make_archive(cfg,{bad:'bad'}))
            stage=root/(str(abs(hash(bad)))+'-stage')
            with self.assertRaises(ValueError):
                updater.stage_archive(archive,stage,cfg['required'])
            self.assertFalse(stage.exists())

if __name__=='__main__':
    unittest.main()
