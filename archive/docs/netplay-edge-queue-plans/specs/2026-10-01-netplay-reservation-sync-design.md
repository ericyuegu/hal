# Netplay reservation sync

Transport update: the approved [protocol-4 design](../../../deploy/netplay/free-tier-design.md)
replaces the external two-second reports with a shared host WebSocket and local
relay. Continuous-session ownership and menu semantics below still apply.
See [the current runbook](../../../deploy/netplay/README.md) for operation.

Date: 2026-10-01. Status: design approved in conversation; awaiting written-spec review.

## Goal

A reservation connects one player to one HAL bot in Slippi direct mode. Today the
website (Worker), the runner, and Slippi coordinate through edge-triggered transitions
(`connecting`, `playing`, `finish-game`, `rematch_wait` → Rematch → `rematch_ready`, …).
Each side must guess what the other believes, and a lost or repeated call needs
special handling. This design replaces that with one owner per fact, level-triggered
state, and a single explicit end command.

Success means:

- The bot stays connected to the player in Slippi after each game until the player
  leaves, disconnects, or a timeout ends the reservation. There is no set and no
  game cap.
- The player can change HAL's character, player identity, stage, and play style from
  the page at any time. A change is never lost; it takes effect at the next
  opportunity, and the page says when.
- A lost, duplicated, or reordered message between the Worker and the runner cannot
  make them disagree.
- Viewers who are not in the queue do not reach the queue's Durable Object.

## Decisions

| Topic | Decision |
|---|---|
| Sync model | Player-owned `settings` with a revision; runner-owned `observed` snapshots with a sequence number; one explicit `end`. |
| Runner channel | Reports every ~2 s; each response carries the latest desired state. The live-settings WebSocket is removed. |
| Slot fairness | No limit while the queue is empty. After 15 min assigned with a non-empty queue, the Worker sets `wind_down`; the runner ends after the current game. |
| Lock-in | The bot hovers its character for 5 s at character select, restarts the hold on each change of character, identity, or stage (30 s cap), then locks in. The page can request an immediate lock. |
| Interrupted games | A game interrupted by a service failure is not recorded as a win or loss. The forfeit credit is removed. |
| Runner failure | Lease expiry, session end, or a retryable runner failure requeues the reservation at the front once; a second failure ends it. |
| Player presence | Kept from the presence change: a queued reservation whose page has not polled for 120 s ends as `player_left`. |

## Slippi behavior this depends on

Verified on 2026-10-01 with two local Slippi 3.6.4 Dolphins (HAL#9000 and CRYO#610),
screenshots, and raw menu events.

1. After a game, both players return to the online character select and stay
   connected. The previous character stays selected but unlocked ("Press START to
   lock in"). The game starts when both players have locked in with Start.
2. Slippi 3.6.4 has no un-ready. `HandleInputsOnCSS` handles lock-in (Start) and
   disconnect (hold Z for 48 frames) only; `PreventAPressCharUnselect` and
   `PreventBPressCharUnselect` block unselecting.
3. The bot cannot see the player's lock-in: its menu event bytes do not change.
4. A disconnect at character select is not visible in the menu event either. After a
   disconnect, Start opens code entry (`submenu == NAME_ENTRY_SUBMENU`); while
   connected, Start locks in or does nothing. Start is therefore a connection probe.
5. If the player's client dies mid-game, the bot's game continues for about 30 s, then
   ends as `NO_CONTEST`, and the bot is left disconnected at character select.
6. L+R+A+Start works in direct mode (pause is enabled). It ends the game as
   `NO_CONTEST` and both players stay connected.
7. While a game is paused, Slippi emits no frames.
8. Game 1's stage is random. After that, the loser of the previous game picks.
9. Z on the stage-select screen toggles Frozen Pokémon Stadium. The toggle persists for
   the Dolphin session. libmelee's `MenuHelper` assumes "not frozen" when it is
   constructed, so a fresh helper flips a frozen toggle back to normal (observed:
   game 2 "Frozen", game 3 "Normal" with the same picker). The current runner creates
   a fresh helper for every rematch and has this bug.

## Architecture

### Ownership

| Fact | Owner | Readers |
|---|---|---|
| Reservation exists, who holds it, when it ended and why | Worker | page, runner |
| What the player wants: `settings` | Player's page (through the Worker) | runner |
| `wind_down`, lock requests | Worker (player request or yield policy) | runner |
| What is happening in Slippi: `observed` | Runner | page |

No side writes a fact it does not own. Every message carries the writer's complete
current view of what it owns, so the newest message wins and older ones are harmless.

### Worker lifecycle

A reservation has three statuses:

- `queued`: waiting for a slot.
- `assigned`: a runner slot holds it under a lease.
- `ended`: terminal, with an `end_reason`.

Finer states (`booting`, `waiting_for_player`, `character_select`, `in_game`,
`paused`) are the runner's `observed.phase`. The Worker never enforces transitions
between phases.

| End reason | Decided by | When |
|---|---|---|
| `player_canceled` | page | Leave while queued; or `wind_down = player` reached a safe point |
| `player_left` | Worker | queued and the page has not polled for 120 s |
| `player_disconnected` | runner | the Start probe found the player gone |
| `no_show` | runner | the player did not connect within 60 s |
| `idle_timeout` | runner | 5 min at character select without a game, or 60 s paused |
| `yielded` | runner | `wind_down = yield` reached a safe point |
| `service_failure` | runner or Worker | a failure with no retry left |

A safe point is any moment outside `in_game` and `paused`.

Runner failure, one rule: when a lease expires, a runner session ends, or the runner
sends `end(service_failure, retryable=true)`, the Worker requeues the reservation at
the front if `attempt < 2`, clearing `observed`; otherwise it ends it as
`service_failure`. Presence ends a requeued reservation whose player has gone.

Yield: on every report and alarm, if a reservation has been assigned for at least
15 min and the queue is non-empty, the Worker sets `wind_down = yield`. It never clears
it.

Lease: each accepted report renews the lease. The lease is 20 s, or 60 s while
`observed.phase` is `in_game` or `paused`, so a short network outage during a game
does not end it.

### Documents

`settings`, written only through the page's routes:

```json
{
  "revision": 3,
  "character": "FALCO",
  "imitation": "MANG#0",
  "stage": "POKEMON_STADIUM",
  "desired_return": 20.0,
  "temperature": 1.0
}
```

- `revision` starts at 1 on create and increases by one on every accepted change.
- `stage` is a policy stage, or `null` for random. HAL uses it only when HAL picks the
  stage.
- `character`, `imitation`, and `stage` take effect at the next lock-in.
  `desired_return` and `temperature` take effect immediately, including mid-game.

`observed`, written only by the runner:

```json
{
  "seq": 41,
  "phase": "character_select",
  "phase_seconds_left": 3.2,
  "bot_code": "HAL#9000",
  "seen_revision": 3,
  "locked_revision": 2,
  "finished_games": [
    { "number": 1, "stage": "FINAL_DESTINATION", "result": "win" },
    { "number": 2, "stage": "POKEMON_STADIUM", "result": "no_contest" }
  ]
}
```

- `seq` increases with every report within one attempt. The Worker keeps a snapshot
  only if its `seq` is greater than the stored one.
- `phase_seconds_left` is the runner's remaining time for the phase deadline: connect
  (60 s), lock-in hold (5 s, 30 s cap), idle (5 min), or pause (60 s). The Worker
  converts it to an absolute `phase_deadline` with its own clock, so the page needs no
  runner clock.
- `seen_revision` is the newest settings revision the runner has received; play style
  is in effect from that revision. `locked_revision` is the revision used for the
  current or last lock-in; character, identity, and stage are in effect from that
  revision.
- `finished_games` is the complete list every time. Results are from the human's side:
  `win`, `loss`, or `no_contest`. The Worker stores each game once by number; a
  different result for a stored number is a 409. A game interrupted by a failure is
  never listed.

### Routes

Player:

| Route | Effect |
|---|---|
| `POST /v1/jobs` | Create with `settings` revision 1. Unchanged validation, rate limit, and capacity checks. |
| `GET /v1/jobs/:id` | Return `{id, status, end_reason, queue_position, settings, observed, phase_deadline, wind_down, lock_requests}`. Refreshes presence at most every 10 s. |
| `PATCH /v1/jobs/:id/settings` | Merge any of the settings fields and bump `revision`. Allowed while `queued` or `assigned`. Replaces `PATCH /policy` and `POST /rematch`. |
| `POST /v1/jobs/:id/lock` | Increment `lock_requests`. Allowed while `assigned`. |
| `DELETE /v1/jobs/:id` | `queued`: end as `player_canceled`. `assigned`: set `wind_down = player`. |

Runner (protocol version 3):

| Route | Effect |
|---|---|
| `POST /v1/runner/sessions/:id/claim` | Unchanged; the job body now includes `settings`. |
| `POST /v1/runner/jobs/:id/report` | Body: `observed`. Renews the lease; stores the snapshot if `seq` is newer; records new finished games. Response: `{status, end_reason, settings, wind_down, lock_requests}`. 409 when the slot does not hold the job. |
| `POST /v1/runner/jobs/:id/end` | Body: `{reason, retryable}`. Idempotent: an ended reservation returns its current state. Applies the requeue rule for `service_failure`. |
| `POST /v1/runner/jobs/:id/replay` | Unchanged; keyed by game number. |
| `GET /v1/runner/jobs/:id` | Unchanged read for the holder. |

Removed: runner `connecting`, `playing`, `finish-game`, `no-show`, `no-contest`,
`fail`, `forfeit`, `heartbeat`, and `live` (WebSocket); player `/policy` and
`/rematch`. The session, status, claim, pairing, drain, admin, and replay routes stay.

Version bumps: runner protocol 2 → 3; store schema 4 → 5 on a fresh Durable Object
instance (`global-v5`). The runner and the Worker deploy together.

### Runner

Two threads per reservation:

- **Reporter.** Every 2 s it sends the current `observed` snapshot and stores the
  response as the latest desired state. If the response says `ended`, or a report
  returns 409, it signals the main loop to stop. It replaces the heartbeat thread and
  the live-settings socket.
- **Main loop.** It drives Dolphin and reads only the latest desired state. It never
  waits on the network.

Main loop:

1. `booting`: claim, launch Dolphin.
2. `waiting_for_player`: direct-connect to the player's code with the latest
   settings. No connection in 60 s → `end(no_show)`. A `wind_down` aborts the
   connect → `end(player_canceled)`.
3. `character_select`: hover the latest character. Lock in with Start when the hold
   expires (5 s after arriving or after the last change of `character`, `imitation`,
   or `stage`; 30 s cap) or when `lock_requests` increases. Record `locked_revision`.
   Start while unlocked locks in, so the probe cannot run during the hold: the lock-in
   press is the first probe, and after it the bot presses Start every 2 s. Code entry →
   `end(player_disconnected)`. A disconnect during the hold is found at lock-in.
   5 min without a game → `end(idle_timeout)`. `wind_down` → end with its reason.
4. Stage select, when HAL picks: `settings.stage`, or random when `null`. Frozen
   Stadium is selected through the session's tracked toggle state.
5. `in_game`: play with the latest `desired_return` and `temperature`. At the end of
   the game, append to `finished_games`, upload the replay as today, then go to step 3.
   `NO_CONTEST` records `no_contest`; whether the player is still connected is decided
   by step 3's probe.
6. `paused`: no frames for more than 2 s during a game. Report `paused`. 60 s without
   frames → `end(idle_timeout)`. This replaces the 10 s frame-stall failure in game.
7. Errors → `end(service_failure, retryable)` with today's retryable classification.

`hal/sim/netplay.py` gains deterministic menu operations, each unit-tested against
fake menu states:

- `hover_character(character, costume)`: move to and select a character without
  locking in.
- `lock_in()`: press Start once.
- `probe_connection() -> bool`: press Start and report whether code entry opened.
- Frozen Stadium state owned by `NetplaySession` for the life of the Dolphin process
  and carried into every `MenuHelper` it creates.

`start_match` and `start_rematch` stay for ranked play (`hal/eval/ranked.py`). The
Frozen Stadium fix applies to both paths.

### Page

The page polls `GET /v1/jobs/:id` every second and renders from `status` and
`observed.phase`:

| State | Page |
|---|---|
| `queued` | Position in line |
| `booting` | "Starting HAL" |
| `waiting_for_player` | Bot code, connect steps, 60 s countdown |
| `character_select` | "Pick your character in Slippi", hold countdown, **Lock in now** |
| `in_game` | Game number |
| `paused` | "Game paused" with the 60 s countdown |
| `ended` | One line per `end_reason` |

The sentence (player identity, character, stage, difficulty) is editable while
`queued` or `assigned`. A change to character, identity, or stage shows "applies next
game" until `locked_revision` reaches it. A difficulty change shows "in effect" once
`seen_revision` reaches it. The page lists finished games (stage, result). One Leave
button: while queued it leaves; while assigned it reads "Stop after this game" during
a game and "Leave" otherwise. `RematchForm` and the rematch countdown are removed.

### Caching

The stateless Worker serves `GET /v1/options` and `GET /v1/capacity` through the Cache
API (`caches.default`), keyed by URL:

- `/v1/capacity`: 3 s.
- `/v1/options`: 10 s. A policy publish reaches viewers within 10 s; job creation
  still validates against the live policy.

Only 200 responses are cached. The stored copy carries `Cache-Control: max-age`; the
response to the browser keeps `Cache-Control: no-store`. Under load, one request per
colo per TTL reaches the Durable Object instead of one per viewer per poll.

## Failure handling

| Failure | Result |
|---|---|
| Report lost or repeated | Next report carries the full snapshot; stale `seq` ignored. |
| Settings change while the runner is offline | Delivered in the first report response after reconnect. |
| Runner process dies | Lease expires (20 s, or 60 s in game); requeue rule. |
| Runner host session ends | Same requeue rule for every job it held. |
| Player closes the page while queued | `player_left` after 120 s. |
| Player closes the page while assigned | No effect; Slippi presence decides. |
| Player disconnects in Slippi | Start probe after lock-in (within the hold's 30 s cap plus 2 s); mid-game after ~30 s via `NO_CONTEST`, then the probe. |
| Player pauses | Up to 60 s tolerated, then `idle_timeout`. |
| Hung Dolphin in game | Detected after 60 s as a pause expiry. Accepted cost of tolerating pauses. |

## Testing

- **Worker** (`web/netplay-api`, vitest): store tests for the lifecycle, report
  ordering by `seq`, finished-game idempotency, the requeue rule, yield, lease length by
  phase, and presence; route tests for every route above; cost tests for report and
  poll reads and writes; a cache test for `/v1/options` and `/v1/capacity`. The 18
  golden transcripts encode the removed transition API; they are replaced by scenario
  tests of the new lifecycle.
- **Runner** (pytest): unit tests for the reporter, lock-in hold and cap, the
  connection probe, pause handling, finished-game bookkeeping, and the requeue
  classification, using a fake session and a fake queue. Contract tests for the
  protocol-3 client.
- **Menu operations** (pytest): `hover_character`, `lock_in`, `probe_connection`,
  and Frozen Stadium tracking against fake menu states.
- **Integration** (`-m integration`, `HAL_REQUIRE_INTEGRATION=1`): a two-Dolphin
  direct-mode test ported from the probe in `tests/`. It checks consecutive games
  without reconnecting, the lock-in hold, Frozen Stadium on two consecutive Stadium picks
  by the same side, L+R+A+Start continuing the session, and a disconnect detected by the
  probe. It needs two distinct Slippi accounts and skips nothing silently.
- **Page** (`web/netplay`): typecheck, lint, format; manual pass through the debug
  console.
- `scripts/qualify_netplay_059.py` moves to the new player routes.

## Out of scope

- Ranked play keeps `start_rematch`; only the Frozen Stadium fix touches it.
- The live-settings latency of up to ~2 s is accepted.
- Upstreaming the Frozen Stadium fix to the libmelee fork is a follow-up.
