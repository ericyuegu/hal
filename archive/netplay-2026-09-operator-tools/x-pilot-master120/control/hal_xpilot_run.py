"""Run the prepared x_pilot schedule and retain verified Slippi recordings."""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import httpx
import melee
import peppi_py
from botocore.config import Config
from peppi_py.game import EndMethod

import hal_xpilot_chat as chat
import hal_credentials
from hal.data.index import extract_index_entry
from hal.eval.matchups import matchups_for_vs_cpu

ROOT = Path('/home/ericgu/src/hal')
RUN = ROOT / 'runs/netplay/x-pilot-master120'
POLICY_SHA = '0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec'
TERMINAL = {'complete', 'failed', 'canceled', 'no_show'}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
    temporary.replace(path)


def event(kind: str, **values: object) -> None:
    payload = {'at': datetime.now(UTC).isoformat(), 'kind': kind, **values}
    with (RUN / 'events.jsonl').open('a') as output:
        output.write(json.dumps(payload) + '\n')
    print(json.dumps(payload), flush=True)


def bot_reply(after: float, matches, timeout: float = 45) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in chat.CHAT_LOG.read_text().splitlines():
            packet = json.loads(line)
            if packet['observed_at'] < after or packet.get('kind') != 'chat':
                continue
            if packet.get('login') == 'x_pilot_bot' and matches(packet['text']):
                return packet
        time.sleep(1)
    raise TimeoutError('x_pilot did not confirm the requested command')


def choose_agent(agent: str) -> list[dict]:
    before = time.time()
    stopped = chat.send('!stop')
    bot_reply(before, lambda text: text in ('Stopped playing against hal_20xx', "hal_20xx, you're not playing right now."))
    time.sleep(2)
    before = time.time()
    selected = chat.send('!agent ' + agent)
    confirmation = bot_reply(before, lambda text: text == 'hal_20xx has selected ' + agent)
    return [stopped, selected, confirmation]


def queue_request(client: httpx.Client, method: str, creds: dict) -> dict:
    response = client.request(method, '/v1/jobs/' + creds['id'], headers={'Authorization': 'Bearer ' + creds['token']})
    response.raise_for_status()
    return response.json()


def new_game(client: httpx.Client, row: dict) -> tuple[dict, list[dict]]:
    messages = choose_agent(row['twitch_agent'])
    response = client.post('/v1/jobs', json={
        'player_code': 'PHAI#591', 'character': row['hal_character'],
        'imitation': 'MASTER', 'desired_return': 120, 'temperature': 1.0, 'online_delay': 2})
    response.raise_for_status()
    created = response.json()
    creds = {'id': created['id'], 'token': created['token'], 'created_at': time.time()}
    write_json(RUN / 'private' / f"{row['schedule_index']:03d}-{creds['id']}.json", creds)
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = queue_request(client, 'GET', creds)
            if job['status'] == 'connecting':
                if job['connect_code'] != 'HAL#9000':
                    raise RuntimeError('The reservation did not use HAL#9000')
                before = time.time()
                messages.append(chat.send('!play HAL#9000'))
                messages.append(bot_reply(before, lambda text: text.startswith('Connecting to hal_20xx (HAL#9000) ')))
                return creds, messages
            if job['status'] in TERMINAL:
                raise RuntimeError('Reservation ended before connecting: ' + job['status'])
            time.sleep(2)
        raise TimeoutError('HAL reservation did not connect')
    except (RuntimeError, TimeoutError, httpx.HTTPError):
        queue_request(client, 'DELETE', creds)
        raise


def finished_game(client: httpx.Client, creds: dict, row: dict) -> dict:
    deadline = time.monotonic() + 1500
    previous = None
    while time.monotonic() < deadline:
        job = queue_request(client, 'GET', creds)
        if (job['character'], job['imitation'], job['desired_return'], job['temperature'], job['online_delay']) != (row['hal_character'], 'MASTER', 120, 1, 2):
            raise RuntimeError('Reservation conditioning differs from the requested experiment')
        if job['status'] != previous:
            event('job_status', row=row['schedule_index'], job=creds['id'], status=job['status'])
            previous = job['status']
        if job['game_count'] == 1 and job['status'] in ('rematch_wait', 'complete'):
            return job
        if job['status'] in TERMINAL:
            raise RuntimeError('Game did not complete: ' + job['status'] + ' ' + str(job['error_code']))
        time.sleep(2)
    raise TimeoutError('Game did not finish within 25 minutes')


def replay_store():
    values = {}
    for line in hal_credentials.load('runner-env').decode().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            values[key] = value.strip().strip('"').strip("'")
    client = boto3.client('s3', endpoint_url=values['AWS_ENDPOINT_URL'],
        aws_access_key_id=values['AWS_ACCESS_KEY_ID'], aws_secret_access_key=values['AWS_SECRET_ACCESS_KEY'],
        region_name='auto', config=Config(signature_version='s3v4'))
    return client, values['AWS_BUCKET']


def record_game(remote, bucket: str, creds: dict, row: dict, job: dict, messages: list[dict]) -> dict:
    started = datetime.fromtimestamp(creds['created_at'], UTC)
    dates = {started.strftime('%Y/%m/%d'), (started + timedelta(days=1)).strftime('%Y/%m/%d')}
    deadline = time.monotonic() + 180
    key = None
    while time.monotonic() < deadline:
        found = []
        for date in sorted(dates):
            prefix = f"netplay/v1/replays/{date}/PHAI#591/{creds['id']}/"
            listing = remote.list_objects_v2(Bucket=bucket, Prefix=prefix)
            found.extend(item['Key'] for item in listing.get('Contents', []) if item['Key'].endswith('/game-01.json'))
        if len(found) > 1:
            raise RuntimeError('Multiple replay metadata objects for the same game')
        if found:
            key = found[0]
            break
        time.sleep(3)
    if key is None:
        raise TimeoutError('Completed replay metadata did not reach R2')
    directory = RUN / f"{row['schedule_index']:03d}-{creds['id']}"
    directory.mkdir(exist_ok=True)
    metadata_path = directory / 'game-01.json'
    remote.download_file(bucket, key, str(metadata_path))
    metadata = json.loads(metadata_path.read_text())
    if metadata['schema_version'] != 1 or metadata['reservation_id'] != creds['id'] or metadata['game_number'] != 1:
        raise RuntimeError('Replay metadata has the wrong identity')
    if metadata['player_code'] != 'PHAI#591' or metadata['policy_sha256'] != POLICY_SHA:
        raise RuntimeError('Replay metadata has the wrong player or checkpoint')
    replay_key = key.removesuffix('.json') + '.slp'
    if metadata['replay_key'] != replay_key:
        raise RuntimeError('Unexpected replay object key')
    replay_path = directory / 'game-01.slp'
    remote.download_file(bucket, replay_key, str(replay_path))
    digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    if digest != metadata['replay_sha256'] or replay_path.stat().st_size != metadata['replay_size']:
        raise RuntimeError('Downloaded replay does not match its size and hash')
    entry = extract_index_entry(replay_path, compute_sha1=False, with_stats=False)
    if entry is None or entry.outcome is None or entry.outcome.end_method not in (EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED):
        raise RuntimeError('Replay lacks a completed game-end record')
    game = peppi_py.read_slippi(str(replay_path))
    first_frame = next(i for i, frame in enumerate(game.frames.id.to_pylist()) if frame >= 0)
    players = {player.code: player for player in entry.players}
    bot_code = job['connect_code']
    if set(players) != {bot_code, 'PHAI#591'}:
        raise RuntimeError('Unexpected players in replay')
    for code, expected in ((bot_code, row['hal_character']), ('PHAI#591', row['opponent_character'])):
        port = players[code].port
        index = next(i for i, player in enumerate(game.start.players) if int(player.port) + 1 == port)
        actual = game.frames.ports[index].leader.post.character[first_frame].as_py()
        if actual != melee.Character[expected].value:
            raise RuntimeError(f'Replay character mismatch for {code}: {actual}, expected {expected}')
    if metadata['result'] != job['last_result'] or metadata['actual_stage'] != job['actual_stage']:
        raise RuntimeError('Queue result differs from replay metadata')
    return {**row, 'status': 'verified', 'job_id': creds['id'], 'game_number': 1,
        'imitation': 'MASTER', 'desired_return': 120, 'online_delay': 2, 'temperature': 1.0,
        'hal_result': {'win': 'loss', 'loss': 'win', 'tie': 'tie'}[metadata['result']],
        'metadata': metadata, 'local_replay': str(replay_path), 'chat': messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-job', type=Path, required=True)
    parser.add_argument('--max-new-games', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'private').mkdir(exist_ok=True, mode=0o700)
    raw = list(csv.DictReader((ROOT / 'deploy/netplay/x-pilot-96.csv').open()))
    schedule = []
    for index, (row, pair) in enumerate(zip(raw, matchups_for_vs_cpu(96), strict=True), 1):
        if (row['hal_character'], row['opponent_character']) != tuple(char.name for char in pair):
            raise RuntimeError('CSV differs from the canonical matchup schedule')
        row['schedule_index'] = index
        if row['status'] != 'unsupported':
            schedule.append(row)
    remote, bucket = replay_store()
    completed = 0
    try:
        with httpx.Client(base_url='https://20xx.xyz', timeout=20) as client:
            for row in schedule:
                result_path = RUN / f"result-{row['schedule_index']:03d}.json"
                if result_path.exists():
                    if json.loads(result_path.read_text())['status'] != 'verified':
                        raise RuntimeError('Existing result has not been verified')
                    continue
                if (RUN / 'stop-after-game').exists():
                    event('stopped_at_boundary', next_row=row['schedule_index'])
                    break
                creds = None
                try:
                    if row['schedule_index'] == 1:
                        creds = json.loads(args.first_job.read_text())
                        messages = creds.get('chat', [])
                    else:
                        creds, messages = new_game(client, row)
                    write_json(RUN / 'active.json', {'row': row['schedule_index'], 'job_id': creds['id']})
                    job = finished_game(client, creds, row)
                    record = record_game(remote, bucket, creds, row, job, messages)
                    write_json(result_path, record)
                    event('verified', row=row['schedule_index'], hal=row['hal_character'], agent=row['twitch_agent'], result=record['hal_result'], replay=record['local_replay'])
                    completed += 1
                finally:
                    if creds is not None:
                        queue_request(client, 'DELETE', creds)
                time.sleep(2)
                if args.max_new_games is not None and completed >= args.max_new_games:
                    break
            chat.send('!stop')
            event('run_finished', verified=len(list(RUN.glob('result-*.json'))))
    finally:
        remote.close()


if __name__ == '__main__':
    main()
