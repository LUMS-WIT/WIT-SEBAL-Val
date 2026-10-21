"""Release export must exclude local data and session history by construction."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from modules.source_export import run


class SourceExportTests(unittest.TestCase):
    def test_only_reviewed_source_is_exported_and_existing_export_is_protected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            for name in ['main.py', 'config.py', 'config_local.example.py', 'modules/analysis.py',
                         'config_local.py', '.git/config', 'dataset/observations.csv',
                         'outputs/chat.md', 'private_archive/conversation.txt',
                         'extra code/SEBAL/pysebal_py3.py', 'extra code/SEBAL/private.xlsx']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('example')
            target = Path(directory) / 'public'
            run(root, target)
            exported = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
            self.assertEqual(exported, {'main.py', 'config.py', 'config_local.example.py',
                                       'modules/analysis.py', 'extra code/SEBAL/pysebal_py3.py', 'SOURCE_MANIFEST.json'})
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                run(root, target)


if __name__ == '__main__':
    unittest.main()
