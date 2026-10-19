import json
import os
from pathlib import Path

def read_records(path):
    """Read committed newline-terminated records after an interrupted process.

    Only an incomplete final line is ignored. Corrupt complete lines fail closed.
    Closing an append flushes Python buffers; no power-loss/fsync claim is made.
    """
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            yield json.loads(line)

class JsonlJournal:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.count = 0

    def __len__(self):
        return self.count

    def __iter__(self):
        return read_records(self.path)

    def append(self, record):
        data = (json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        with self.path.open('ab') as stream:
            stream.write(data)
        self.count += 1

    def materialize(self):
        """Stream a final array in O(n) time and bounded additional memory."""
        target = self.path.with_suffix('.json')
        temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write('[')
            for (index, row) in enumerate(self):
                if index:
                    stream.write(',')
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
            stream.write(']\n')
        temporary.replace(target)
        return target
