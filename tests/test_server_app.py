import base64
import io
import os
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient


os.environ.setdefault('APP_PASSWORD', 'test-password')

SERVER_DIR = Path(__file__).resolve().parents[1] / 'server'
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import server as speech_server


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.deleted = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}


def auth_headers(username='alice', password='test-password'):
    token = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('ascii')
    return {'Authorization': f'Basic {token}'}


class ServerAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(speech_server.app)

    def setUp(self):
        self.original_password = speech_server._APP_PASSWORD
        self.original_db_path = speech_server._publish_store.db_path
        self.original_s3_client = speech_server._publish_store._s3_client
        self.tempdir = tempfile.TemporaryDirectory()
        speech_server._APP_PASSWORD = 'test-password'
        speech_server._publish_store.db_path = str(Path(self.tempdir.name) / 'published.db')
        speech_server._publish_store._s3_client = FakeS3()
        speech_server._publish_store.init_db()

    def tearDown(self):
        speech_server._APP_PASSWORD = self.original_password
        speech_server._publish_store.db_path = self.original_db_path
        speech_server._publish_store._s3_client = self.original_s3_client
        self.tempdir.cleanup()

    def publish_sample(self, username='alice', user_id='user-1'):
        metadata = {
            'user_id': user_id,
            'username': username,
            'filename': 'sample.wav',
            'metrics': {'score': 12},
            'audio_sr': 16000,
            'duration_s': 1.5,
            'version': 'test',
        }
        response = self.client.post(
            '/publish/sustained-phonation',
            headers=auth_headers(username=username),
            data={'metadata': speech_server.json.dumps(metadata)},
            files={'file': ('sample.wav', b'RIFFfakeWAVE', 'audio/wav')},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()['id']

    def test_health_is_public_and_logged(self):
        with self.assertLogs('speech_metrics.server', level='INFO') as logs:
            response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertIn('X-Request-ID', response.headers)
        self.assertIn('path=/health', '\n'.join(logs.output))

    def test_me_requires_auth(self):
        response = self.client.get('/me')
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers['WWW-Authenticate'], 'Basic realm="Speech Assessment"')
        self.assertIn('X-Request-ID', response.headers)

    def test_me_returns_authenticated_username(self):
        response = self.client.get('/me', headers=auth_headers(username='alice'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'username': 'alice'})

    def test_publish_and_list_entries(self):
        entry_id = self.publish_sample()
        self.assertGreater(entry_id, 0)
        response = self.client.get('/published/sustained-phonation', headers=auth_headers(username='alice'))
        self.assertEqual(response.status_code, 200)
        entries = response.json()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['username'], 'alice')
        self.assertEqual(entries[0]['metrics']['score'], 12)

    def test_delete_rejects_wrong_owner(self):
        entry_id = self.publish_sample(username='alice')
        response = self.client.delete(f'/published/{entry_id}', headers=auth_headers(username='bob'))
        self.assertEqual(response.status_code, 403)

    def test_delete_allows_owner(self):
        entry_id = self.publish_sample(username='alice')
        delete_response = self.client.delete(f'/published/{entry_id}', headers=auth_headers(username='alice'))
        self.assertEqual(delete_response.status_code, 200)
        list_response = self.client.get('/published/sustained-phonation', headers=auth_headers(username='alice'))
        self.assertEqual(list_response.json(), [])

    def test_delete_legacy_entry_uses_user_id_fallback(self):
        with closing(speech_server._publish_store.get_db()) as conn:
            conn.execute(
                'INSERT INTO published '
                '(task,user_id,username,filename,published_at,s3_key,metrics,reference_text,audio_sr,duration_s,version) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (1, 'legacy-user', None, 'legacy.wav', '2026-01-01T00:00:00+00:00', 'legacy-key', '{}', None, 16000, 1.0, 'legacy'),
            )
            conn.commit()
            entry_id = conn.execute('SELECT id FROM published').fetchone()['id']
        speech_server._publish_store._s3_client.objects['legacy-key'] = b'audio'
        response = self.client.delete(
            f'/published/{entry_id}?user_id=legacy-user',
            headers=auth_headers(username=''),
        )
        self.assertEqual(response.status_code, 200)

    def test_analyze_sustained_phonation_route(self):
        with mock.patch.object(speech_server, 'wav_bytes_to_sound', return_value='sound'), mock.patch.object(
            speech_server,
            'analyze_task1',
            return_value={'metric': 123},
        ):
            response = self.client.post(
                '/analyze/sustained-phonation',
                headers=auth_headers(username='alice'),
                files={'file': ('sample.wav', b'RIFFfakeWAVE', 'audio/wav')},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['metric'], 123)
        self.assertIn('version', body)


if __name__ == '__main__':
    unittest.main()
