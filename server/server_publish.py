import json
import sqlite3
import urllib.request
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import boto3
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse


TASK_SLUGS = {'sustained-phonation': 1, 'pitch-glides': 2, 'reading-passage': 3}


def detect_aws_region() -> str:
    try:
        token = urllib.request.urlopen(
            urllib.request.Request(
                'http://169.254.169.254/latest/api/token',
                method='PUT',
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '10'},
            ),
            timeout=1,
        ).read().decode()
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/placement/region',
            headers={'X-aws-ec2-metadata-token': token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return boto3.session.Session().region_name or 'us-east-1'


@dataclass
class PublishStore:
    db_path: str
    logger: object
    s3_bucket: str = 'speech-metrics-storage'
    s3_prefix: str = 'public-samples/'
    aws_region: str = field(default_factory=detect_aws_region)
    _s3_client: object | None = None

    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with closing(self.get_db()) as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS published (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    task           INTEGER NOT NULL,
                    user_id        TEXT,
                    username       TEXT,
                    filename       TEXT,
                    published_at   TEXT NOT NULL,
                    s3_key         TEXT NOT NULL,
                    metrics        TEXT,
                    reference_text TEXT,
                    audio_sr       INTEGER,
                    duration_s     REAL,
                    version        TEXT
                )
                '''
            )
            try:
                conn.execute('ALTER TABLE published ADD COLUMN username TEXT')
            except Exception:
                pass
            conn.commit()

    def get_s3(self):
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.aws_region)
        return self._s3_client

    def init(self) -> None:
        self.init_db()
        self.logger.info('S3 configured: %s/%s', self.s3_bucket, self.s3_prefix)


def create_publish_router(
    store: PublishStore,
    request_username: Callable[[Request], str],
    sanitize_username: Callable[[str], str],
) -> APIRouter:
    router = APIRouter()

    @router.post('/publish/{task_slug}')
    async def publish_entry(task_slug: str, request: Request, file: UploadFile = File(...), metadata: str = Form(...)):
        task_n = TASK_SLUGS.get(task_slug)
        if task_n is None:
            raise HTTPException(status_code=404, detail='Unknown task')
        data = json.loads(metadata)
        audio = await file.read()
        s3_key = f'{store.s3_prefix}{task_slug}/{uuid.uuid4()}.wav'
        owner_username = request_username(request) or sanitize_username(str(data.get('username') or ''))
        store.get_s3().put_object(Bucket=store.s3_bucket, Key=s3_key, Body=audio, ContentType='audio/wav')
        now = datetime.now(timezone.utc).isoformat()
        with closing(store.get_db()) as conn:
            cur = conn.execute(
                'INSERT INTO published '
                '(task,user_id,username,filename,published_at,s3_key,metrics,reference_text,audio_sr,duration_s,version) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (
                    task_n,
                    data.get('user_id'),
                    owner_username or None,
                    data.get('filename'),
                    now,
                    s3_key,
                    json.dumps(data.get('metrics', {})),
                    data.get('reference_text'),
                    data.get('audio_sr'),
                    data.get('duration_s'),
                    data.get('version'),
                ),
            )
            conn.commit()
            entry_id = cur.lastrowid
        return {'id': entry_id, 'published_at': now}

    @router.get('/published/{task_slug}')
    def get_published(task_slug: str):
        task_n = TASK_SLUGS.get(task_slug)
        if task_n is None:
            raise HTTPException(status_code=404, detail='Unknown task')
        with closing(store.get_db()) as conn:
            rows = conn.execute(
                'SELECT id,task,user_id,username,filename,published_at,metrics,audio_sr,duration_s,version '
                'FROM published WHERE task=? ORDER BY published_at DESC',
                (task_n,),
            ).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            entry['metrics'] = json.loads(entry['metrics'] or '{}')
            result.append(entry)
        return result

    @router.delete('/published/{entry_id}')
    def delete_published(entry_id: int, request: Request, user_id: Optional[str] = None):
        current_username = request_username(request)
        with closing(store.get_db()) as conn:
            row = conn.execute('SELECT s3_key, username, user_id FROM published WHERE id=?', (entry_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail='Not found')
            owner_username = sanitize_username(row['username'] or '')
            if owner_username and current_username:
                if current_username != owner_username:
                    raise HTTPException(status_code=403, detail='Forbidden')
            elif not row['user_id'] or user_id != row['user_id']:
                raise HTTPException(status_code=403, detail='Forbidden')
            try:
                store.get_s3().delete_object(Bucket=store.s3_bucket, Key=row['s3_key'])
            except Exception as exc:
                store.logger.warning('S3 delete warning: %s', exc)
            conn.execute('DELETE FROM published WHERE id=?', (entry_id,))
            conn.commit()
        return {'ok': True}

    @router.get('/audio/{entry_id}')
    def stream_audio(entry_id: int):
        with closing(store.get_db()) as conn:
            row = conn.execute('SELECT s3_key, filename FROM published WHERE id=?', (entry_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='Not found')
        obj = store.get_s3().get_object(Bucket=store.s3_bucket, Key=row['s3_key'])
        fname = (row['filename'] or 'recording').rstrip('.zip') + '.wav'
        return StreamingResponse(
            obj['Body'],
            media_type='audio/wav',
            headers={'Content-Disposition': f'attachment; filename="{fname}"'},
        )

    return router
