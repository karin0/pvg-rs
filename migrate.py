import os
import sys
import json
import sqlite3

def main():
    json_path = sys.argv[1]
    sqlite_path = sys.argv[2]

    with open(json_path, encoding='utf-8') as fp:
        fav = json.load(fp)

    items = tuple(
        ((d := json.dumps(item, ensure_ascii=False, separators=(',', ':'), indent=None)), d['id'])
        for item in fav
    )
    del fav
    print(f'Loaded {len(items)} items')

    if os.path.exists(sqlite_path):
        raise FileExistsError(sqlite_path)

    with sqlite3.connect(sqlite_path, autocommit=False) as conn:
        with open('schema.sql', encoding='utf-8') as f:
            sql = f.read()

        conn.executescript(sql)
        conn.executemany('INSERT INTO Illust (data, iid) VALUES (?, ?)', items)


if __name__ == '__main__':
    main()
