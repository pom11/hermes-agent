"""A caller-supplied ``dedupe`` token names a route wake's durable admission identity (#152206).

``session_admissions`` is UNIQUE on ``(principal_id, target_session_id, request_id)`` and replays an
exact identity, but ``admit_route_wake`` minted ``json.dumps([producer, key, uuid4().hex])`` per call
— so the fence could never recognise a repeat wake and every producer retry queued another turn.
These pin the three properties of the fix over a real GatewayRunner, DiscordAdapter, SessionAuthority
and SQLite: ``dedupe=None`` keeps the old bytes exactly, a named wake admits once, and a named wake is
still carried across ``/new`` — which only holds while the identity stays a JSON list headed by a
``_ROUTE_WAKE_PRODUCERS`` name (``move_route_wakes`` parses it).
"""
import asyncio
import json

import pytest

from tests.waiting import async_wait_until

TOKEN = "kanban:ops:t_1:" + "d" * 40


async def _busy_session(monkeypatch):
    """A real runner/adapter/authority with one human turn holding the route, blocked until released.

    The committed human turn is what gives a route wake provenance to borrow; holding it open parks
    each wake ``queued`` on the route's FIFO, where its admission row can be read.
    """
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    running, finish = asyncio.Event(), asyncio.Event()
    executed = []

    class Agent:
        def interrupt(self, *args, **kwargs):
            pass

    async def turn(event, source, quick_key, run_generation):
        executed.append((event.text, runner.session_store.peek_session_id(quick_key)))
        if event.text == 'HUMAN':
            runner._session_state(quick_key).turn.agent = Agent()
            running.set()
            await finish.wait()
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='HUMAN', source=source, message_id='HUMAN'))
    await asyncio.wait_for(running.wait(), 10)
    # Admissions are keyed by the session's id; the wake's identity carries its session KEY.
    return runner, adapter, authority, source, key, runner.session_store.peek_session_id(key), executed, finish


def _wake_rows(authority, session_id):
    from hermes_state_runtime import list_session_admissions
    return [r for r in list_session_admissions(authority.db, session_id=str(session_id), pending_only=False)
            if r['principal_id'].startswith('automation:')]


@pytest.mark.asyncio
async def test_dedupe_none_keeps_the_random_uuid_identity(monkeypatch):
    """The default is byte-identical to today: a JSON list headed by the producer, random third
    element — so cron, /loop-tick and goal continuation are untouched until they opt in."""
    runner, adapter, authority, source, key, session_id, executed, finish = await _busy_session(monkeypatch)
    from gateway.session_automation import admit_route_wake

    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake')
    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake')
    tokens = [json.loads(r['request_id']) for r in _wake_rows(authority, session_id)]
    assert [t[0] for t in tokens] == ['kanban-wake', 'kanban-wake']
    assert [t[1] for t in tokens] == [key, key]
    assert len({t[2] for t in tokens}) == 2, 'two unnamed wakes are two identities'
    assert all(len(t[2]) == 32 and int(t[2], 16) >= 0 for t in tokens), 'still uuid4().hex bytes'
    finish.set()
    await async_wait_until(lambda: [t for t, _ in executed].count('WAKE') == 2, timeout=15,
                           what='both unnamed wakes to run')


@pytest.mark.asyncio
async def test_same_dedupe_admits_one_wake_not_two(monkeypatch):
    """One named wake retried is ONE admission: the fence replays it instead of queueing a copy."""
    runner, adapter, authority, source, key, session_id, executed, finish = await _busy_session(monkeypatch)
    from gateway.session_automation import admit_route_wake

    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake', None, TOKEN)
    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake', None, TOKEN)
    rows = _wake_rows(authority, session_id)
    assert len(rows) == 1, f'a named wake retried must replay one admission, got {rows}'
    assert json.loads(rows[0]['request_id']) == ['kanban-wake', key, TOKEN]
    finish.set()
    await async_wait_until(lambda: [t for t, _ in executed].count('WAKE') == 1, timeout=15,
                           what='the named wake to run')
    await async_wait_until(lambda: [r['status'] for r in _wake_rows(authority, session_id)] == ['terminal'],
                           timeout=15, what='the named wake to settle')
    assert len(_wake_rows(authority, session_id)) == 1, 'the replayed identity produced no second admission'
    assert [t for t, _ in executed].count('WAKE') == 1, 'and no second turn'


@pytest.mark.asyncio
async def test_two_event_ids_of_one_task_are_two_wakes(monkeypatch):
    """Two events of the same task in the same second stay two wakes: the identity is the event
    ids, never the task, never a wall-clock window."""
    runner, adapter, authority, source, _key, session_id, executed, finish = await _busy_session(monkeypatch)
    from gateway.session_automation import admit_route_wake

    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake', None, TOKEN)
    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake', None, TOKEN.replace('d' * 40, 'e' * 40))
    assert len(_wake_rows(authority, session_id)) == 2
    finish.set()
    await async_wait_until(lambda: [t for t, _ in executed].count('WAKE') == 2, timeout=15,
                           what='both named wakes to run')


@pytest.mark.asyncio
async def test_named_wake_is_carried_across_new(monkeypatch):
    """A deterministic identity still survives ``/new``: move_route_wakes only carries a wake whose
    request_id parses as a JSON list headed by a route-wake producer, so the token must ride the
    third element — an opaque identity would silently strand the wake on the abandoned session."""
    runner, adapter, authority, source, key, session_id, executed, finish = await _busy_session(monkeypatch)
    from gateway.session_automation import admit_route_wake

    old = session_id  # /new rotates it; the wake must land on the fresh one
    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake', None, TOKEN)
    from gateway.platforms.event import MessageEvent
    await adapter.handle_message(MessageEvent(text='/new', source=source, message_id='new'))
    new = runner.session_store.peek_session_id(key)
    assert new != old
    finish.set()
    await async_wait_until(lambda: any(t == 'WAKE' for t, _ in executed), timeout=15,
                           what='the named wake to run after /new')
    assert [sid for t, sid in executed if t == 'WAKE'] == [new], 'the named wake runs once, on the fresh session'
    assert not [r for r in _wake_rows(authority, old) if r['status'] == 'queued'], 'nothing stranded on the old session'
