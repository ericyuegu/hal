'use client';

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import type { ReactNode } from 'react';
import { Check, Copy, Gamepad2, LoaderCircle, X } from 'lucide-react';

import { Sentence, ShortcutSheet, useHotkeys } from '@/components/sentence';
import type { Panel } from '@/components/sentence';
import {
  ApiError,
  cancelJob,
  Capacity,
  Choice,
  createJob,
  CreateJob,
  fallbackOptions,
  getCapacity,
  getJob,
  getOptions,
  Job,
  Options,
  requestRematch,
  updatePolicy,
} from '@/lib/netplay-api';
import {
  clampReturn,
  defaultImitation,
  toDifficulty,
  toReturn,
} from '@/lib/roster';

type SavedJob = { id: string; token: string };
// Temperature is not exposed; the API applies the policy default.
type Prefs = {
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number;
};

const savedJobKey = 'hal-netplay-job-v1';
// v2 stores raw desired_return and drops temperature.
const prefsKey = 'hal-netplay-prefs-v2';
const terminal = new Set(['complete', 'failed', 'canceled', 'no_show']);
// Mirrors hal/netplay_service/domain.py validate_player_code.
const playerCodePattern = /^[A-Z0-9]{1,8}#[0-9]{1,4}$/;
const defaultCharacter = 'FALCO';
const streamChannel = 'hal_20xx';
const unavailableCapacity: Capacity = {
  capacity: 2,
  healthy_slots: 0,
  active: 0,
  queued: 0,
  service_status: 'unavailable',
  service_message: 'Game servers are unavailable. Try again shortly.',
  target_fps: 60,
  game_fps: null,
  frame_interval_p95_ms: null,
  dolphin_step_p95_ms: null,
  policy_round_trip_p95_ms: null,
  model_inference_p95_ms: null,
  batch_wait_p95_ms: null,
  recoveries: 0,
};

const storageListeners = new Set<() => void>();

function subscribeStorage(listener: () => void) {
  storageListeners.add(listener);
  window.addEventListener('storage', listener);
  return () => {
    storageListeners.delete(listener);
    window.removeEventListener('storage', listener);
  };
}

// Storage can throw in private windows; every reader treats that as empty.
function readRaw(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: unknown) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Device-local convenience only.
  }
  for (const listener of storageListeners) listener();
}

/** Returns undefined during server render, then the validated stored value. */
function useStored<T>(
  key: string,
  valid: (value: unknown) => value is T,
): T | null | undefined {
  const raw = useSyncExternalStore(
    subscribeStorage,
    () => readRaw(key),
    () => undefined,
  );
  return useMemo(() => {
    if (raw === undefined) return undefined;
    if (raw === null) return null;
    try {
      const value: unknown = JSON.parse(raw);
      return valid(value) ? value : null;
    } catch {
      return null;
    }
  }, [raw, valid]);
}

function isSavedJob(value: unknown): value is SavedJob {
  const job = value as Partial<SavedJob> | null;
  return typeof job?.id === 'string' && typeof job.token === 'string';
}

function isPrefs(value: unknown): value is Prefs {
  const prefs = value as Partial<Prefs> | null;
  return (
    typeof prefs?.player_code === 'string' &&
    typeof prefs.character === 'string' &&
    typeof prefs.imitation === 'string' &&
    typeof prefs.online_delay === 'number' &&
    typeof prefs.desired_return === 'number' &&
    Number.isFinite(prefs.desired_return)
  );
}

function errorText(cause: unknown, fallback: string): string {
  return cause instanceof Error ? cause.message : fallback;
}

export default function Home() {
  const [options, setOptions] = useState<Options>(fallbackOptions);
  const [capacity, setCapacity] = useState<Capacity | null>(null);
  const storedJob = useStored(savedJobKey, isSavedJob);
  const storedPrefs = useStored(prefsKey, isPrefs);
  const loaded = storedJob !== undefined;
  const saved = storedJob ?? null;
  const prefs = storedPrefs ?? null;
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const forget = useCallback(() => {
    writeStored(savedJobKey, null);
    setJob(null);
    setError('');
  }, []);

  const join = useCallback(async (values: CreateJob) => {
    setBusy(true);
    setError('');
    requestNotifications();
    try {
      const created = await createJob(values);
      const credentials = { id: created.id, token: created.token };
      const nextPrefs: Prefs = {
        player_code: values.player_code,
        character: values.character,
        imitation: values.imitation,
        online_delay: values.online_delay,
        desired_return: values.desired_return ?? created.desired_return ?? 0,
      };
      writeStored(savedJobKey, credentials);
      writeStored(prefsKey, nextPrefs);
      setJob(created);
      return created;
    } catch (cause) {
      setError(errorText(cause, 'Could not join the queue.'));
      throw cause;
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void getOptions()
      .then(setOptions)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let canceled = false;
    async function refresh() {
      try {
        const next = await getCapacity();
        if (!canceled) setCapacity(next);
      } catch {
        if (!canceled) setCapacity(unavailableCapacity);
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => {
      canceled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!saved) return;
    let canceled = false;
    async function refresh() {
      try {
        const next = await getJob(saved!.id, saved!.token);
        if (!canceled) {
          setJob(next);
          setError('');
        }
      } catch (cause) {
        if (canceled) return;
        // The service no longer knows this reservation; drop the stale credential.
        if (cause instanceof ApiError && [401, 403, 404].includes(cause.status))
          forget();
        else setError(errorText(cause, 'Could not read reservation status.'));
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => {
      canceled = true;
      window.clearInterval(timer);
    };
  }, [saved, forget]);

  useStatusAlerts(job, options);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const registration = context.registerTool(
      {
        name: 'join_hal_netplay_queue',
        title: 'Join HAL netplay queue',
        description:
          'Create one HAL direct-netplay reservation and show its queue status.',
        inputSchema: {
          type: 'object',
          properties: {
            player_code: { type: 'string' },
            character: {
              type: 'string',
              enum: options.characters.map((choice) => choice.value),
            },
            imitation: {
              type: 'string',
              enum: options.imitations.map((choice) => choice.value),
            },
            online_delay: { type: 'integer', enum: options.online_delays },
            desired_return: {
              type: 'number',
              minimum: options.desired_return_range[0],
              maximum: options.desired_return_range[1],
            },
          },
          required: ['player_code', 'character', 'imitation', 'online_delay'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        async execute(input) {
          const values = validateToolInput(input, options);
          const created = await join(values);
          return {
            id: created.id,
            status: created.status,
            queue_position: created.queue_position,
          };
        },
      },
      { signal: lifecycle.signal },
    );
    void Promise.resolve(registration).catch(() => undefined);
    return () => lifecycle.abort();
  }, [join, options]);

  async function cancel() {
    if (!saved) return;
    setBusy(true);
    setError('');
    try {
      setJob(await cancelJob(saved.id, saved.token));
    } catch (cause) {
      setError(errorText(cause, 'Could not cancel the reservation.'));
    } finally {
      setBusy(false);
    }
  }

  let body: ReactNode;
  if (!loaded || (saved && !job)) {
    body = (
      <p className="loading">
        <LoaderCircle className="spin" size={18} /> One moment…
      </p>
    );
  } else if (job && saved) {
    body = (
      <Reservation
        job={job}
        options={options}
        saved={saved}
        busy={busy}
        error={error}
        cancel={cancel}
        forget={forget}
        requeue={() => {
          forget();
          if (prefs) void join(prefs).catch(() => undefined);
        }}
        update={setJob}
        setBusy={setBusy}
        setError={setError}
      />
    );
  } else {
    body = (
      <JoinForm
        options={options}
        capacity={capacity}
        prefs={prefs}
        busy={busy}
        error={error}
        dismissError={() => setError('')}
        join={join}
      />
    );
  }

  return (
    <main className="wrap">
      <Header capacity={capacity} />
      {body}
    </main>
  );
}

function requestNotifications() {
  if (typeof Notification === 'undefined') return;
  if (Notification.permission === 'default')
    void Notification.requestPermission().catch(() => undefined);
}

/** Keeps the tab title current and pings the player when they must act. */
function useStatusAlerts(job: Job | null, options: Options) {
  const previous = useRef<string | null>(null);
  useEffect(() => {
    const status = job?.status ?? null;
    document.title = job ? `${tabTitle(job)} · HAL` : 'HAL';
    const from = previous.current;
    previous.current = status;
    if (!job || from === null || from === status) return;
    const message =
      status === 'connecting'
        ? `Direct-connect to ${job.connect_code ?? 'HAL'} in Slippi.`
        : status === 'rematch_wait'
          ? `Game ${job.game_count} done. Choose a rematch within ${minutes(options.rematch_seconds)}.`
          : null;
    if (
      !message ||
      !document.hidden ||
      typeof Notification === 'undefined' ||
      Notification.permission !== 'granted'
    )
      return;
    new Notification(statusTitle(job), { body: message, tag: 'hal-netplay' });
  }, [job, options.rematch_seconds]);
}

function tabTitle(job: Job): string {
  if (job.status === 'queued') return `#${job.queue_position ?? '–'} in queue`;
  if (job.status === 'connecting') return '● Connect now';
  if (job.status === 'rematch_wait') return '● Rematch?';
  return statusTitle(job);
}

function openSlots(capacity: Capacity): number {
  return Math.max(capacity.healthy_slots - capacity.active, 0);
}

function Header({ capacity }: { capacity: Capacity | null }) {
  const [howTo, setHowTo] = useState(false);
  useEffect(() => {
    if (!howTo) return;
    function outside(event: MouseEvent) {
      if (!(event.target as Element | null)?.closest('.howto-wrap'))
        setHowTo(false);
    }
    document.addEventListener('click', outside);
    return () => document.removeEventListener('click', outside);
  }, [howTo]);

  return (
    <header className="top">
      <div className="logo">
        HAL<i>.</i>
      </div>
      <div className="right">
        <span
          className="status"
          data-status={capacity?.service_status ?? 'loading'}
          title={capacity?.service_message}
        >
          <b aria-hidden="true">●</b>{' '}
          {capacity
            ? `${openSlots(capacity)} open · ${capacity.queued} waiting`
            : 'Connecting…'}
        </span>
        <div className="howto-wrap">
          <button
            type="button"
            className="howto"
            aria-expanded={howTo}
            onClick={() => setHowTo(!howTo)}
          >
            How do I play?
          </button>
          {howTo && (
            <div className="pop">
              <b>HAL is a Melee AI trained on human replays.</b> You play it
              over Slippi netplay.
              <ol>
                <li>Install Slippi and set up netplay.</li>
                <li>Fill in the sentence and press Play.</li>
                <li>When it’s your turn, direct-connect to the code shown.</li>
              </ol>
              <a
                href="https://slippi.gg/netplay"
                target="_blank"
                rel="noopener"
              >
                slippi.gg/netplay →
              </a>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

const joinShortcuts: [ReactNode, string][] = [
  [
    <>
      <kbd>P</kbd>
      <kbd>1</kbd>
      <kbd>/</kbd>
    </>,
    'Pick player',
  ],
  [
    <>
      <kbd>C</kbd>
      <kbd>2</kbd>
    </>,
    'Pick character',
  ],
  [
    <>
      <kbd>D</kbd>
      <kbd>3</kbd>
    </>,
    'Set difficulty',
  ],
  [
    <>
      <kbd>[</kbd>
      <kbd>]</kbd>
    </>,
    'Difficulty −5 / +5',
  ],
  [<kbd key="k">K</kbd>, 'Edit connect code'],
  [<kbd key="enter">↵</kbd>, 'Play (⌘/Ctrl ↵ from a text field)'],
  [<kbd key="esc">Esc</kbd>, 'Close'],
  [<kbd key="help">?</kbd>, 'This sheet'],
];

function JoinForm({
  options,
  capacity,
  prefs,
  busy,
  error,
  dismissError,
  join,
}: {
  options: Options;
  capacity: Capacity | null;
  prefs: Prefs | null;
  busy: boolean;
  error: string;
  dismissError: () => void;
  join: (values: CreateJob) => Promise<Job>;
}) {
  // The last reservation on this device prefills every field.
  const [playerCode, setPlayerCode] = useState(prefs?.player_code ?? '');
  const [touched, setTouched] = useState(false);
  const [character, setCharacter] = useState(
    prefs?.character ?? defaultCharacter,
  );
  const [imitation, setImitation] = useState(
    prefs?.imitation ?? defaultImitation(options.imitations),
  );
  const [delay, setDelay] = useState(prefs?.online_delay ?? 2);
  const [desired, setDesired] = useState(
    prefs?.desired_return ?? options.default_desired_return,
  );
  const [open, setOpen] = useState<Panel | null>(null);
  const [sheet, setSheet] = useState(false);
  const code = useRef<HTMLInputElement>(null);

  // Stored choices can outlive what the current policy offers.
  const characterValue = has(options.characters, character)
    ? character
    : has(options.characters, defaultCharacter)
      ? defaultCharacter
      : options.characters[0].value;
  const imitationValue = has(options.imitations, imitation)
    ? imitation
    : defaultImitation(options.imitations);
  const delayValue = options.online_delays.includes(delay)
    ? delay
    : options.online_delays[0];
  const range = options.desired_return_range;
  const desiredValue = clampReturn(desired, range);
  const difficulty = toDifficulty(desiredValue, range);

  const codeValid = playerCodePattern.test(playerCode);
  const showCodeError = touched && !codeValid;
  const unavailable = capacity !== null && capacity.healthy_slots === 0;

  function setDifficulty(value: number) {
    setDesired(toReturn(Math.max(0, Math.min(100, value)), range));
  }

  function submit() {
    setTouched(true);
    if (!codeValid) {
      setOpen(null);
      code.current?.focus();
      return;
    }
    if (busy || unavailable) return;
    setOpen(null);
    void join({
      player_code: playerCode,
      character: characterValue,
      imitation: imitationValue,
      online_delay: delayValue,
      desired_return: desiredValue,
    }).catch(() => undefined);
  }

  useHotkeys({
    p: () => setOpen('player'),
    '1': () => setOpen('player'),
    '/': () => setOpen('player'),
    c: () => setOpen('char'),
    '2': () => setOpen('char'),
    d: () => setOpen('diff'),
    '3': () => setOpen('diff'),
    '[': () => setDifficulty(difficulty - 5),
    ']': () => setDifficulty(difficulty + 5),
    k: () => {
      setOpen(null);
      code.current?.focus();
      code.current?.select();
    },
    '?': () => setSheet(!sheet),
    enter: submit,
    'mod+enter': submit,
    escape: () => (sheet ? setSheet(false) : setOpen(null)),
  });

  return (
    <>
      <Sentence
        lead="I want to play"
        options={options}
        imitation={{ value: imitationValue, set: setImitation }}
        character={{ value: characterValue, set: setCharacter }}
        difficulty={{ value: difficulty, set: setDifficulty }}
        open={open}
        setOpen={setOpen}
      />
      <div className="bottom">
        <div className="field">
          <label className="lbl" htmlFor="code">
            Your connect code
          </label>
          <input
            ref={code}
            id="code"
            className="code"
            placeholder="CODE#123"
            autoComplete="off"
            autoCapitalize="characters"
            spellCheck={false}
            maxLength={13}
            value={playerCode}
            aria-invalid={showCodeError || undefined}
            aria-describedby={showCodeError ? 'code-hint' : undefined}
            onBlur={() => setTouched(playerCode !== '')}
            onChange={(event) =>
              setPlayerCode(
                event.target.value
                  .toUpperCase()
                  .replaceAll('＃', '#')
                  .replace(/\s/g, ''),
              )
            }
          />
          {showCodeError && (
            <p className="hint bad" id="code-hint">
              Use the format CODE#123, exactly as Slippi shows it.
            </p>
          )}
          <details className="adv">
            <summary>Advanced</summary>
            <div className="advrow">
              <span>
                Frame delay{' '}
                <span className="seg">
                  {options.online_delays.map((frames) => (
                    <button
                      key={frames}
                      type="button"
                      aria-pressed={frames === delayValue}
                      onClick={() => setDelay(frames)}
                    >
                      {frames}f
                    </button>
                  ))}
                </span>
              </span>
            </div>
          </details>
        </div>
        <div className="go">
          <button
            type="button"
            className="play"
            title="Enter"
            disabled={busy || unavailable}
            onClick={submit}
          >
            {busy ? 'Joining…' : 'Play →'}
          </button>
          <div className="line">
            {queueLine(capacity, options)}
            <span className="kbtn-sep"> · </span>
            <button
              type="button"
              className="kbtn"
              onClick={() => setSheet(true)}
            >
              <kbd>?</kbd> shortcuts
            </button>
          </div>
          <StreamNotice />
        </div>
      </div>
      {unavailable && <p className="notice">{capacity.service_message}</p>}
      <ErrorMessage value={error} dismiss={dismissError} />
      {sheet && (
        <ShortcutSheet
          shortcuts={joinShortcuts}
          close={() => setSheet(false)}
        />
      )}
    </>
  );
}

function queueLine(capacity: Capacity | null, options: Options): string {
  const games = `up to ${options.max_games} games`;
  if (capacity === null) return `Checking the queue · ${games}`;
  if (capacity.healthy_slots === 0) return `Servers unavailable · ${games}`;
  if (capacity.queued === 0 && openSlots(capacity) > 0)
    return `You’re first in line · ${games}`;
  return `${capacity.queued} waiting · ${games}`;
}

function StreamNotice() {
  return (
    <p className="line">
      Games may be streamed live on{' '}
      <a
        href={`https://www.twitch.tv/${streamChannel}`}
        target="_blank"
        rel="noopener"
      >
        twitch.tv/{streamChannel}
      </a>
      .
    </p>
  );
}

/** Difficulty edits apply mid-game without an Apply button. */
function useLiveDifficulty(
  job: Job,
  saved: SavedJob,
  options: Options,
  update: (job: Job) => void,
  setError: (value: string) => void,
) {
  const range = options.desired_return_range;
  // Only a user edit is sent; the job's own value is never echoed back.
  const [edit, setEdit] = useState<number | null>(null);
  const [applied, setApplied] = useState(false);
  const desired = edit ?? job.desired_return ?? options.default_desired_return;

  useEffect(() => {
    if (edit === null || edit === job.desired_return) return;
    const timer = window.setTimeout(() => {
      updatePolicy(saved.id, saved.token, { desired_return: edit })
        .then((next) => {
          update(next);
          setApplied(true);
        })
        .catch((cause: unknown) =>
          setError(errorText(cause, 'Could not update the difficulty.')),
        );
    }, 400);
    return () => window.clearTimeout(timer);
  }, [edit, job.desired_return, saved, update, setError]);

  useEffect(() => {
    if (!applied) return;
    const timer = window.setTimeout(() => setApplied(false), 2000);
    return () => window.clearTimeout(timer);
  }, [applied]);

  const difficulty = toDifficulty(clampReturn(desired, range), range);
  return {
    difficulty,
    setDifficulty: (value: number) =>
      setEdit(toReturn(Math.max(0, Math.min(100, value)), range)),
    applied,
  };
}

function Reservation({
  job,
  options,
  saved,
  busy,
  error,
  cancel,
  forget,
  requeue,
  update,
  setBusy,
  setError,
}: {
  job: Job;
  options: Options;
  saved: SavedJob;
  busy: boolean;
  error: string;
  cancel: () => Promise<void>;
  forget: () => void;
  requeue: () => void;
  update: (job: Job) => void;
  setBusy: (value: boolean) => void;
  setError: (value: string) => void;
}) {
  const ended = terminal.has(job.status);
  const [open, setOpen] = useState<Panel | null>(null);
  const live = useLiveDifficulty(job, saved, options, update, setError);
  const rematch = job.status === 'rematch_wait';

  useHotkeys({
    d: ended ? undefined : () => setOpen('diff'),
    '3': ended ? undefined : () => setOpen('diff'),
    '[': ended ? undefined : () => live.setDifficulty(live.difficulty - 5),
    ']': ended ? undefined : () => live.setDifficulty(live.difficulty + 5),
    p: rematch ? () => setOpen('player') : undefined,
    '1': rematch ? () => setOpen('player') : undefined,
    c: rematch ? () => setOpen('char') : undefined,
    '2': rematch ? () => setOpen('char') : undefined,
    escape: () => setOpen(null),
  });

  return (
    <>
      <p className="eyebrow">{statusTitle(job)}</p>
      <Sentence
        small
        lead={ended ? 'You played' : 'You’re playing'}
        options={options}
        imitation={{ value: job.imitation }}
        character={{ value: job.character }}
        difficulty={{
          value: live.difficulty,
          set: ended ? undefined : live.setDifficulty,
        }}
        open={open}
        setOpen={setOpen}
      />
      <div className="tags">
        <span className="tag mono">{job.player_code}</span>
        <span className="tag">{job.online_delay}f delay</span>
        {!ended && (
          <span className="tag quiet" aria-live="polite">
            {live.applied ? (
              <>
                <Check size={12} /> Applied at the next replan
              </>
            ) : (
              'Difficulty applies mid-game'
            )}
          </span>
        )}
      </div>
      <div className="card">
        <Progress job={job} options={options} />
        <div className="status-panel" data-status={job.status}>
          <StatusBody job={job} options={options} />
        </div>
        {rematch && (
          <RematchForm
            key={job.game_count}
            job={job}
            saved={saved}
            options={options}
            busy={busy}
            open={open}
            setOpen={setOpen}
            cancel={cancel}
            update={update}
            setBusy={setBusy}
            setError={setError}
          />
        )}
        <div className="card-footer">
          <p className="muted">
            {ended
              ? `${job.game_count} of ${options.max_games} games played`
              : `Game ${Math.min(job.game_count + 1, options.max_games)} of up to ${options.max_games}`}
          </p>
          {ended ? (
            <div className="actions">
              <button type="button" className="ghost" onClick={forget}>
                Change settings
              </button>
              <button
                type="button"
                className="play small"
                onClick={requeue}
                disabled={busy}
              >
                Queue again →
              </button>
            </div>
          ) : (
            !rematch && (
              <button
                type="button"
                className="ghost"
                onClick={() => void cancel()}
                disabled={busy || job.cancel_after_game}
              >
                {cancelLabel(job)}
              </button>
            )
          )}
        </div>
      </div>
      <ErrorMessage value={error} dismiss={() => setError('')} />
    </>
  );
}

function cancelLabel(job: Job): string {
  if (job.cancel_after_game) return 'Ending after this game';
  if (job.status === 'playing') return 'Stop after this game';
  if (job.status === 'queued') return 'Leave queue';
  return 'Cancel';
}

const steps = [
  { label: 'Queue', statuses: ['queued', 'leased'] },
  { label: 'Connect', statuses: ['connecting'] },
  { label: 'Play', statuses: ['playing', 'rematch_wait', 'rematch_ready'] },
  { label: 'Done', statuses: ['complete', 'failed', 'canceled', 'no_show'] },
];

function Progress({ job, options }: { job: Job; options: Options }) {
  const current = steps.findIndex((step) => step.statuses.includes(job.status));
  return (
    <ol className="progress" aria-label="Reservation progress">
      {steps.map((step, index) => (
        <li
          key={step.label}
          data-state={
            index < current ? 'done' : index === current ? 'current' : 'todo'
          }
          aria-current={index === current ? 'step' : undefined}
        >
          {step.label === 'Play' && job.game_count > 0
            ? `Game ${Math.min(job.game_count + (job.status === 'rematch_wait' ? 0 : 1), options.max_games)}`
            : step.label}
        </li>
      ))}
    </ol>
  );
}

function StatusBody({ job, options }: { job: Job; options: Options }) {
  if (job.status === 'queued') {
    return (
      <BigStatus
        value={job.queue_position === null ? '—' : `#${job.queue_position}`}
        label={job.queue_position === 1 ? 'You’re next' : 'Place in queue'}
        detail="Keep this tab open. We’ll notify you when your slot is ready."
      />
    );
  }
  if (job.status === 'leased') {
    return (
      <BigStatus
        icon={<LoaderCircle className="spin" size={40} />}
        label="Booting HAL’s Dolphin"
        detail="Your slot is reserved. Open Slippi so you’re ready to connect."
      />
    );
  }
  if (job.status === 'connecting') {
    return (
      <div>
        <p className="lbl">Direct-connect to</p>
        <CopyCode code={job.connect_code} />
        <ol className="connect-steps">
          <li>
            In Slippi, open <b>Online → Direct</b>.
          </li>
          <li>Enter the code above and pick your character.</li>
          <li>HAL joins automatically.</li>
        </ol>
        <Countdown
          deadline={job.connect_deadline}
          total={options.no_show_seconds}
          suffix="left to connect"
        />
      </div>
    );
  }
  if (job.status === 'playing') {
    return (
      <BigStatus
        icon={<Gamepad2 size={40} />}
        label={`Game ${job.game_count + 1} in progress`}
        detail={
          job.cancel_after_game
            ? 'The set ends after this game.'
            : 'This page updates when the game ends.'
        }
      />
    );
  }
  if (job.status === 'rematch_wait') {
    return (
      <div>
        <BigStatus
          value={resultText(job.last_result)}
          label={
            job.actual_stage
              ? `Game ${job.game_count} · ${label(options.stages, job.actual_stage)}`
              : `Game ${job.game_count} complete`
          }
          detail="Stay on the Slippi results screen while you choose."
        />
        <Countdown
          deadline={job.rematch_deadline}
          total={options.rematch_seconds}
          suffix="to start the next game"
        />
      </div>
    );
  }
  if (job.status === 'rematch_ready') {
    return (
      <BigStatus
        icon={<LoaderCircle className="spin" size={40} />}
        label="Setting up the rematch"
        detail="Keep the Slippi connection open."
      />
    );
  }
  return (
    <BigStatus
      icon={job.status === 'complete' ? <Check size={40} /> : <X size={40} />}
      label={statusTitle(job)}
      detail={terminalDetail(job)}
    />
  );
}

// last_result is from the human player's perspective.
function resultText(result: string | null): string {
  if (result === 'win') return 'You won';
  if (result === 'loss') return 'HAL won';
  if (result === 'tie') return 'Tie';
  return 'Game over';
}

function terminalDetail(job: Job): string {
  if (job.status === 'no_show')
    return 'HAL waited but no connection arrived. Queue again when you are ready.';
  if (job.error_code) return `Reason: ${pretty(job.error_code)}.`;
  if (job.status === 'complete') return 'Thanks for playing.';
  return 'Queue again whenever you like.';
}

function CopyCode({ code }: { code: string | null }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return (
    <button
      type="button"
      className="copy-code"
      disabled={!code}
      onClick={() => {
        if (!code) return;
        void navigator.clipboard
          .writeText(code)
          .then(() => setCopied(true))
          .catch(() => undefined);
      }}
      aria-label={code ? `Copy connect code ${code}` : 'Connect code pending'}
    >
      <span className="copy-code-value">{code ?? '—'}</span>
      <span className="copy-code-action" aria-live="polite">
        {copied ? (
          <>
            <Check size={16} /> Copied
          </>
        ) : (
          <>
            <Copy size={16} /> Copy
          </>
        )}
      </span>
    </button>
  );
}

function RematchForm({
  job,
  saved,
  options,
  busy,
  open,
  setOpen,
  cancel,
  update,
  setBusy,
  setError,
}: {
  job: Job;
  saved: SavedJob;
  options: Options;
  busy: boolean;
  open: Panel | null;
  setOpen: (panel: Panel | null) => void;
  cancel: () => Promise<void>;
  update: (job: Job) => void;
  setBusy: (value: boolean) => void;
  setError: (value: string) => void;
}) {
  const [character, setCharacter] = useState(job.character);
  const [imitation, setImitation] = useState(job.imitation);
  const [stage, setStage] = useState(
    job.requested_stage ?? job.actual_stage ?? 'BATTLEFIELD',
  );

  async function submit() {
    setBusy(true);
    setError('');
    try {
      update(
        await requestRematch(saved.id, saved.token, {
          character,
          imitation,
          stage,
        }),
      );
    } catch (cause) {
      setError(errorText(cause, 'Could not request the rematch.'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rematch">
      <Sentence
        small
        lead="Next, I want to play"
        options={options}
        imitation={{ value: imitation, set: setImitation }}
        character={{ value: character, set: setCharacter }}
        stage={{ value: stage, set: setStage }}
        open={open}
        setOpen={setOpen}
      />
      <p className="hint">
        HAL picks the stage only if it lost the last game. If you lost, pick it
        in Slippi.
      </p>
      <div className="actions">
        <button
          type="button"
          className="ghost"
          disabled={busy}
          onClick={() => void cancel()}
        >
          End set
        </button>
        <button
          type="button"
          className="play small"
          disabled={busy}
          onClick={() => void submit()}
        >
          {busy ? 'Starting…' : `Play game ${job.game_count + 1} →`}
        </button>
      </div>
    </div>
  );
}

function BigStatus({
  value,
  icon,
  label: title,
  detail,
}: {
  value?: string;
  icon?: ReactNode;
  label: string;
  detail: ReactNode;
}) {
  return (
    <div>
      <div className="status-value">{value ?? icon}</div>
      <h2 className="status-label">{title}</h2>
      <p className="muted">{detail}</p>
    </div>
  );
}

function Countdown({
  deadline,
  total,
  suffix,
}: {
  deadline: number | null;
  total: number;
  suffix: string;
}) {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => window.clearInterval(timer);
  }, []);
  const seconds =
    deadline === null || now === null
      ? null
      : Math.max(0, Math.ceil(deadline - now));
  const fraction = seconds === null ? 1 : Math.min(seconds / total, 1);
  return (
    <div className="countdown" data-urgent={seconds !== null && seconds <= 60}>
      <div className="countdown-head">
        <span className="countdown-clock">
          {seconds === null ? '–:––' : clock(seconds)}
        </span>
        <span className="muted">{suffix}</span>
      </div>
      <div className="meter" aria-hidden="true">
        <span style={{ transform: `scaleX(${fraction})` }} />
      </div>
    </div>
  );
}

function ErrorMessage({
  value,
  dismiss,
}: {
  value: string;
  dismiss: () => void;
}) {
  return value ? (
    <div role="alert" className="error">
      <span>{value}</span>
      <button type="button" onClick={dismiss} aria-label="Dismiss error">
        <X size={16} />
      </button>
    </div>
  ) : null;
}

function statusTitle(job: Job): string {
  return (
    {
      queued: 'In the queue',
      leased: 'Starting HAL',
      connecting: 'Your slot is ready',
      playing: 'Game on',
      rematch_wait: 'Run it back?',
      rematch_ready: 'Preparing rematch',
      complete: 'Set complete',
      failed: 'Something went wrong',
      canceled: 'Reservation ended',
      no_show: 'Connection timed out',
    }[job.status] ?? pretty(job.status)
  );
}

function has(choices: Choice[], value: string): boolean {
  return choices.some((choice) => choice.value === value);
}

function label(choices: Choice[], value: string): string {
  return (
    choices.find((choice) => choice.value === value)?.label ?? pretty(value)
  );
}

function pretty(value: string): string {
  return value
    .toLowerCase()
    .replaceAll('_', ' ')
    .replace(/^./, (letter) => letter.toUpperCase());
}

function clock(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

function minutes(seconds: number): string {
  return seconds % 60 === 0 && seconds >= 60
    ? `${seconds / 60} minute${seconds === 60 ? '' : 's'}`
    : `${seconds} seconds`;
}

function validateToolInput(input: unknown, options: Options): CreateJob {
  if (typeof input !== 'object' || input === null)
    throw new Error('Queue input must be an object.');
  const values = input as Record<string, unknown>;
  const strings = ['player_code', 'character', 'imitation'] as const;
  for (const name of strings)
    if (typeof values[name] !== 'string' || !values[name])
      throw new Error(`${name} must be a non-empty string.`);
  if (!has(options.characters, values.character as string))
    throw new Error('character is not available.');
  if (!has(options.imitations, values.imitation as string))
    throw new Error('imitation is not available.');
  if (!options.online_delays.includes(values.online_delay as number))
    throw new Error('online_delay is not available.');
  const desired = values.desired_return;
  if (
    desired !== undefined &&
    (typeof desired !== 'number' ||
      !Number.isFinite(desired) ||
      desired < options.desired_return_range[0] ||
      desired > options.desired_return_range[1])
  )
    throw new Error('desired_return is out of range.');
  return {
    player_code: values.player_code as string,
    character: values.character as string,
    imitation: values.imitation as string,
    online_delay: values.online_delay as number,
    desired_return:
      desired === undefined ? options.default_desired_return : desired,
  };
}
