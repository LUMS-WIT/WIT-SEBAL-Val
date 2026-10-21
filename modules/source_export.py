"""Create a reviewed, source-only folder without private files or Git history."""
from pathlib import Path
import hashlib
import json
import shutil


def source_files(root):
    root = Path(root)
    names = ['.gitignore', 'README.md', 'main.py', 'config.py', 'config_local.example.py',
             'requirements.yml', 'utils.py', 'metrics.py', 'scaling.py', 'sms_calibration.py',
             'validation.drawio', 'LICENSE', 'CITATION.cff']
    files = [root / name for name in names if (root / name).is_file()]
    for folder, pattern in [('modules', '*.py'), ('workflows', '*.py'), ('tests', '*.py'),
                            ('docs', '*.md'), ('calibrations', '*.js'), ('calibrations', '*.drawio'),
                            ('extra code/SEBAL', '*.py'), ('extra code/SEBAL', '*.md')]:
        files.extend((root / folder).glob(pattern))
    # Symlinks can otherwise turn a public-looking filename into a private file.
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in files):
        raise ValueError('Source export refuses symlinks or files outside the repository.')
    return sorted(set(files))


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('Choose a separate export directory, not the repository or its parent.')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Export destination must be empty; retain or move the previous export: {output}')
    files = source_files(root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for source in files:
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(f'Exported {len(files)} source files, without data, local configuration or Git history: {output}')
    return output
