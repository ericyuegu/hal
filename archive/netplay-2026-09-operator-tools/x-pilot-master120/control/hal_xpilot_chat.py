"""Operate the owner's authorized x_pilot chat session through Twitch APIs."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import httpx
from websockets.sync.client import connect

import hal_credentials

CHAT_LOG = Path('/home/ericgu/src/hal/runs/netplay/x-pilot-master120/chat.jsonl')


def auth_headers(force_refresh: bool = False) -> dict[str, str]:
    with (hal_credentials.DIRECTORY / 'twitch-chat.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        auth = json.loads(hal_credentials.load('twitch-chat'))
        if force_refresh or time.time() >= auth['obtained_at'] + auth['expires_in'] - 300:
            with httpx.Client(timeout=20) as client:
                response = client.post('https://id.twitch.tv/oauth2/token', data={
                    'client_id': auth['client_id'], 'grant_type': 'refresh_token',
                    'refresh_token': auth['refresh_token']})
                response.raise_for_status()
            refreshed = response.json()
            if not {'access_token', 'refresh_token', 'expires_in'} <= refreshed.keys():
                raise RuntimeError('Twitch returned an incomplete refresh response')
            auth = {**refreshed, 'client_id': auth['client_id'], 'obtained_at': time.time()}
            hal_credentials.save('twitch-chat', json.dumps(auth).encode())
    return {'Authorization': 'Bearer ' + auth['access_token'], 'Client-Id': auth['client_id']}


def identity(client: httpx.Client) -> dict:
    response = client.get('https://id.twitch.tv/oauth2/validate', headers={
        'Authorization': auth_headers()['Authorization'].replace('Bearer ', 'OAuth ', 1)})
    if response.status_code == 401:
        response = client.get('https://id.twitch.tv/oauth2/validate', headers={
            'Authorization': auth_headers(force_refresh=True)['Authorization'].replace('Bearer ', 'OAuth ', 1)})
    response.raise_for_status()
    data = response.json()
    if data['login'] != 'hal_20xx' or data['client_id'] != '8dcduusqdvjllf8xvtadn153o7z9tc':
        raise RuntimeError('Unexpected Twitch account or application')
    if not {'user:read:chat', 'user:write:chat'} <= set(data['scopes']):
        raise RuntimeError('Twitch token lacks requested chat scopes')
    return data


def channel(client: httpx.Client) -> dict:
    response = client.get('https://api.twitch.tv/helix/users', params={'login': 'x_pilot'}, headers=auth_headers())
    response.raise_for_status()
    users = response.json()['data']
    if len(users) != 1 or users[0]['login'] != 'x_pilot':
        raise RuntimeError('Could not identify the x_pilot channel')
    return users[0]


def log(payload: dict) -> None:
    record = {'observed_at': time.time(), **payload}
    with CHAT_LOG.open('a') as output:
        output.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def send(message: str) -> dict:
    if message not in ('!play HAL#9000', '!stop', '!status', '!help'):
        agents = 'cptfalcon falco fox jigglypuff luigi marth peach pikachu popo samus sheik yoshi'.split()
        if message not in {'!agent gm-v2-' + char for char in agents}:
            raise ValueError('Command is outside the requested x_pilot run')
    with httpx.Client(timeout=20) as client:
        user = identity(client)
        target = channel(client)
        response = client.post('https://api.twitch.tv/helix/chat/messages', headers=auth_headers(), json={
            'broadcaster_id': target['id'], 'sender_id': user['user_id'], 'message': message})
        response.raise_for_status()
        result = response.json()['data'][0]
        log({'kind': 'sent', 'login': user['login'], 'text': message, **result})
        if result['is_sent'] is not True:
            raise RuntimeError('Twitch rejected chat message: ' + json.dumps(result['drop_reason']))
        return result


def listen() -> None:
    with httpx.Client(timeout=20) as client:
        user = identity(client)
        target = channel(client)
        response = client.get('https://api.twitch.tv/helix/streams', params={'user_id': target['id']}, headers=auth_headers())
        response.raise_for_status()
        log({'kind': 'channel', 'login': user['login'], 'channel_id': target['id'], 'live': bool(response.json()['data'])})
        url = 'wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=30'
        resumed = False
        while True:
            with connect(url, proxy=None, open_timeout=20, ping_interval=None) as socket:
                welcome = json.loads(socket.recv(timeout=15))
                if welcome['metadata']['message_type'] != 'session_welcome':
                    raise RuntimeError('Expected Twitch EventSub welcome')
                if not resumed:
                    response = client.post('https://api.twitch.tv/helix/eventsub/subscriptions', headers=auth_headers(), json={
                        'type': 'channel.chat.message', 'version': '1',
                        'condition': {'broadcaster_user_id': target['id'], 'user_id': user['user_id']},
                        'transport': {'method': 'websocket', 'session_id': welcome['payload']['session']['id']}})
                    response.raise_for_status()
                    log({'kind': 'subscribed', 'status': response.json()['data'][0]['status']})
                while True:
                    packet = json.loads(socket.recv(timeout=45))
                    kind = packet['metadata']['message_type']
                    if kind == 'notification':
                        event = packet['payload']['event']
                        if event['chatter_user_login'] == 'hal_20xx' or (
                            event['chatter_user_login'] == 'x_pilot_bot' and 'hal_20xx' in event['message']['text']
                        ):
                            log({'kind': 'chat', 'login': event['chatter_user_login'], 'text': event['message']['text'], 'message_id': event['message_id']})
                    elif kind == 'session_reconnect':
                        url = packet['payload']['session']['reconnect_url']
                        if not url.startswith('wss://eventsub.wss.twitch.tv/'):
                            raise RuntimeError('Unexpected Twitch reconnect host')
                        resumed = True
                        break
                    elif kind != 'session_keepalive':
                        raise RuntimeError('Unexpected Twitch event: ' + kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('listen', 'send'))
    parser.add_argument('message', nargs='?')
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == 'send':
        if args.message is None:
            parser.error('send requires a message')
        send(args.message)
    else:
        listen()


if __name__ == '__main__':
    main()
