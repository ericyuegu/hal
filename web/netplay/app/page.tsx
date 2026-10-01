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
  Capacity,
  Choice,
  createJob,
  CreateJob,
  EndReason,
  fallbackOptions,
  getCapacity,
  getJob,
  getOptions,
  Job,
  leaveJob,
  Options,
  requestLock,
  Settings,
  SettingsUpdate,
  updateSettings,
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
        desired_return:
          values.desired_return ?? created.settings.desired_return ?? 0,
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

  useStatusAlerts(job);

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

  async function leave() {
    if (!saved) return;
    setBusy(true);
    setError('');
    try {
      setJob(await leaveJob(saved.id, saved.token));
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
        leave={leave}
        forget={forget}
        requeue={() => {
          forget();
          if (prefs)
            void join({ ...prefs, stage: null }).catch(() => undefined);
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
function useStatusAlerts(job: Job | null) {
  const previous = useRef<string | null>(null);
  useEffect(() => {
    const status = job
      ? `${job.status}:${job.observed?.phase ?? 'unobserved'}`
      : null;
    document.title = job ? `${tabTitle(job)} · HAL` : 'HAL';
    const from = previous.current;
    previous.current = status;
    if (!job || from === null || from === status) return;
    const message =
      job.observed?.phase === 'waiting_for_player'
        ? `Direct-connect to ${job.observed.bot_code ?? 'HAL'} in Slippi.`
        : job.observed?.phase === 'character_select'
          ? 'Pick your character in Slippi.'
          : null;
    if (
      !message ||
      !document.hidden ||
      typeof Notification === 'undefined' ||
      Notification.permission !== 'granted'
    )
      return;
    new Notification(statusTitle(job), { body: message, tag: 'hal-netplay' });
  }, [job]);
}

function tabTitle(job: Job): string {
  if (job.status === 'queued') return `#${job.queue_position ?? '–'} in queue`;
  if (job.observed?.phase === 'waiting_for_player') return '● Connect now';
  if (job.observed?.phase === 'character_select') return 'Pick your character';
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
      stage: null,
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
        lead="I want HAL to play like"
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
            onKeyDown={(event) => {
              // Mod+Enter already reaches the page hotkey.
              if (event.key !== 'Enter' || event.metaKey || event.ctrlKey)
                return;
              event.preventDefault();
              submit();
            }}
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
            {queueLine(capacity)}
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

function queueLine(capacity: Capacity | null): string {
  if (capacity === null) return 'Checking the queue';
  if (capacity.healthy_slots === 0) return 'Servers unavailable';
  if (capacity.queued === 0 && openSlots(capacity) > 0)
    return 'You’re first in line';
  return `${capacity.queued} waiting`;
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

type EditableField = 'character' | 'imitation' | 'stage' | 'desired_return';
type SettingsChange = { revision: number; kind: 'selection' | 'difficulty' };

function useSettingsEditor(
  job: Job,
  saved: SavedJob,
  update: (job: Job) => void,
  setError: (value: string) => void,
) {
  const [draft, setDraft] = useState<SettingsUpdate>({});
  const [sending, setSending] = useState(false);
  const [change, setChange] = useState<SettingsChange | null>(null);

  useEffect(() => {
    if (sending) return;
    const values = Object.fromEntries(
      Object.entries(draft).filter(
        ([field, value]) => value !== job.settings[field as keyof Settings],
      ),
    ) as SettingsUpdate;
    if (Object.keys(values).length === 0) {
      if (Object.keys(draft).length === 0) return;
      const cleanup = window.setTimeout(() => setDraft({}), 0);
      return () => window.clearTimeout(cleanup);
    }
    const timer = window.setTimeout(() => {
      setSending(true);
      updateSettings(saved.id, saved.token, values)
        .then((next) => {
          update(next);
          const selection = ['character', 'imitation', 'stage'].some((field) =>
            Object.hasOwn(values, field),
          );
          setChange({
            revision: next.settings.revision,
            kind: selection ? 'selection' : 'difficulty',
          });
          setDraft((current) => {
            const remaining = Object.entries(current).filter(
              ([field, value]) =>
                !Object.hasOwn(values, field) ||
                values[field as keyof SettingsUpdate] !== value,
            );
            return Object.fromEntries(remaining) as SettingsUpdate;
          });
        })
        .catch((cause: unknown) => {
          setDraft({});
          setError(errorText(cause, 'Could not update HAL’s settings.'));
        })
        .finally(() => setSending(false));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [draft, job.settings, saved, sending, setError, update]);

  function value<Field extends EditableField>(field: Field): Settings[Field] {
    return (
      Object.hasOwn(draft, field) ? draft[field] : job.settings[field]
    ) as Settings[Field];
  }

  function set<Field extends EditableField>(
    field: Field,
    next: Settings[Field],
  ) {
    setDraft((current) => ({ ...current, [field]: next }));
  }

  return {
    value,
    set,
    pending: sending || Object.keys(draft).length > 0,
    change,
  };
}

function Reservation({
  job,
  options,
  saved,
  busy,
  error,
  leave,
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
  leave: () => Promise<void>;
  forget: () => void;
  requeue: () => void;
  update: (job: Job) => void;
  setBusy: (value: boolean) => void;
  setError: (value: string) => void;
}) {
  const ended = job.status === 'ended';
  const [open, setOpen] = useState<Panel | null>(null);
  const editor = useSettingsEditor(job, saved, update, setError);
  const range = options.desired_return_range;
  const difficulty = toDifficulty(
    clampReturn(
      editor.value('desired_return') ?? options.default_desired_return,
      range,
    ),
    range,
  );

  function setDifficulty(value: number) {
    editor.set(
      'desired_return',
      toReturn(Math.max(0, Math.min(100, value)), range),
    );
  }

  async function lockNow() {
    setBusy(true);
    setError('');
    try {
      update(await requestLock(saved.id, saved.token));
    } catch (cause) {
      setError(errorText(cause, 'Could not lock in HAL.'));
    } finally {
      setBusy(false);
    }
  }

  useHotkeys({
    d: ended ? undefined : () => setOpen('diff'),
    '3': ended ? undefined : () => setOpen('diff'),
    '[': ended ? undefined : () => setDifficulty(difficulty - 5),
    ']': ended ? undefined : () => setDifficulty(difficulty + 5),
    p: ended ? undefined : () => setOpen('player'),
    '1': ended ? undefined : () => setOpen('player'),
    c: ended ? undefined : () => setOpen('char'),
    '2': ended ? undefined : () => setOpen('char'),
    escape: () => setOpen(null),
  });

  return (
    <>
      <p className="eyebrow">{statusTitle(job)}</p>
      <Sentence
        small
        lead={ended ? 'HAL played like' : 'HAL plays like'}
        options={options}
        imitation={{
          value: editor.value('imitation'),
          set: ended ? undefined : (value) => editor.set('imitation', value),
        }}
        character={{
          value: editor.value('character'),
          set: ended ? undefined : (value) => editor.set('character', value),
        }}
        difficulty={{
          value: difficulty,
          set: ended ? undefined : setDifficulty,
        }}
        stage={{
          value: editor.value('stage'),
          set: ended ? undefined : (value) => editor.set('stage', value),
        }}
        open={open}
        setOpen={setOpen}
      />
      <div className="tags">
        <span className="tag mono">{job.player_code}</span>
        <span className="tag">{job.online_delay}f delay</span>
        {!ended && (
          <span className="tag quiet" aria-live="polite">
            {settingsStatus(job, editor.pending, editor.change)}
          </span>
        )}
      </div>
      <div className="card">
        <Progress job={job} />
        <div className="status-panel" data-status={job.status}>
          <StatusBody job={job} busy={busy} lockNow={lockNow} />
        </div>
        <GameList games={job.games} options={options} />
        <div className="card-footer">
          <p className="muted">
            {job.games.length === 0
              ? 'No finished games yet'
              : `${job.games.length} game${job.games.length === 1 ? '' : 's'} finished`}
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
            <button
              type="button"
              className="ghost"
              onClick={() => void leave()}
              disabled={busy || job.wind_down === 'player'}
            >
              {leaveLabel(job)}
            </button>
          )}
        </div>
      </div>
      <ErrorMessage value={error} dismiss={() => setError('')} />
    </>
  );
}

function leaveLabel(job: Job): string {
  if (job.wind_down === 'player') return 'Ending after this game';
  if (job.status === 'queued') return 'Leave queue';
  if (job.observed?.phase === 'in_game' || job.observed?.phase === 'paused')
    return 'Stop after this game';
  return 'Leave';
}

const steps = [
  { label: 'Queue', states: ['queued', 'booting'] },
  { label: 'Connect', states: ['waiting_for_player'] },
  { label: 'Play', states: ['character_select', 'in_game', 'paused'] },
  { label: 'Done', states: ['ended'] },
];

function Progress({ job }: { job: Job }) {
  const state =
    job.status === 'ended'
      ? 'ended'
      : job.status === 'queued'
        ? 'queued'
        : (job.observed?.phase ?? 'booting');
  const current = steps.findIndex((step) => step.states.includes(state));
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
          {step.label === 'Play' && job.games.length > 0
            ? `Game ${job.games.length + 1}`
            : step.label}
        </li>
      ))}
    </ol>
  );
}

function StatusBody({
  job,
  busy,
  lockNow,
}: {
  job: Job;
  busy: boolean;
  lockNow: () => Promise<void>;
}) {
  if (job.status === 'queued') {
    return (
      <BigStatus
        value={job.queue_position === null ? '—' : `#${job.queue_position}`}
        label={job.queue_position === 1 ? 'You’re next' : 'Place in queue'}
        detail="Keep this tab open. We’ll notify you when your slot is ready."
      />
    );
  }
  if (job.status === 'ended') {
    return (
      <BigStatus
        icon={<X size={40} />}
        label={statusTitle(job)}
        detail={endText(job.end_reason)}
      />
    );
  }
  const observed = job.observed;
  if (observed === null || observed.phase === 'booting') {
    return (
      <BigStatus
        icon={<LoaderCircle className="spin" size={40} />}
        label="Starting HAL"
        detail="Your slot is reserved. Open Slippi so you are ready to connect."
      />
    );
  }
  if (observed.phase === 'waiting_for_player') {
    return (
      <div>
        <p className="lbl">Direct-connect to</p>
        <CopyCode code={observed.bot_code} />
        <ol className="connect-steps">
          <li>
            In Slippi, open <b>Online → Direct</b>.
          </li>
          <li>Enter the code above and pick your character.</li>
          <li>HAL joins automatically.</li>
        </ol>
        <Countdown
          deadline={job.phase_deadline}
          total={60}
          suffix="left to connect"
        />
      </div>
    );
  }
  if (observed.phase === 'character_select') {
    if (observed.locked_revision !== null) {
      return (
        <BigStatus
          icon={<Gamepad2 size={40} />}
          label="HAL is locked in"
          detail="Lock in on Slippi to start."
        />
      );
    }
    return (
      <div>
        <BigStatus
          icon={<Gamepad2 size={40} />}
          label="Pick your character in Slippi"
          detail="HAL waits briefly for your selection before it locks in."
        />
        <Countdown
          deadline={job.phase_deadline}
          total={30}
          suffix="until HAL locks in"
        />
        <button
          type="button"
          className="play small lock-now"
          disabled={busy}
          onClick={() => void lockNow()}
        >
          Lock in now
        </button>
      </div>
    );
  }
  if (observed.phase === 'in_game') {
    return (
      <BigStatus
        icon={<Gamepad2 size={40} />}
        label={`Game ${job.games.length + 1}`}
        detail={
          job.wind_down === 'player'
            ? 'The session ends after this game.'
            : 'Playing now.'
        }
      />
    );
  }
  if (observed.phase === 'paused') {
    return (
      <div>
        <BigStatus
          icon={<Gamepad2 size={40} />}
          label="Game paused"
          detail="Resume in Slippi to continue."
        />
        <Countdown
          deadline={job.phase_deadline}
          total={60}
          suffix="until the session ends"
        />
      </div>
    );
  }
  return null;
}

function GameList({
  games,
  options,
}: {
  games: Job['games'];
  options: Options;
}) {
  if (games.length === 0) return null;
  return (
    <ol className="game-list" aria-label="Finished games">
      {games.map((game) => (
        <li key={game.number}>
          <span>Game {game.number}</span>
          <span>{label(options.stages, game.stage)}</span>
          <strong>{resultText(game.result)}</strong>
        </li>
      ))}
    </ol>
  );
}

function resultText(result: Job['games'][number]['result']): string {
  if (result === 'win') return 'You won';
  if (result === 'loss') return 'HAL won';
  return 'No contest';
}

function settingsStatus(
  job: Job,
  pending: boolean,
  change: SettingsChange | null,
): string {
  if (pending) return 'Saving…';
  if (change?.kind === 'selection') {
    if (
      job.observed?.phase === 'character_select' &&
      job.observed.locked_revision === null
    )
      return 'Applies to this game';
    if ((job.observed?.locked_revision ?? 0) >= change.revision)
      return 'In effect';
    return 'Applies next game';
  }
  const revision = change?.revision ?? job.settings.revision;
  return (job.observed?.seen_revision ?? 0) >= revision
    ? 'In effect'
    : 'Applies shortly';
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
  if (job.status === 'queued') return 'In the queue';
  if (job.status === 'ended') return 'Session ended';
  return (
    {
      booting: 'Starting HAL',
      waiting_for_player: 'Your slot is ready',
      character_select: 'Choose your character',
      in_game: 'Game on',
      paused: 'Game paused',
    }[job.observed?.phase ?? 'booting'] ?? 'Starting HAL'
  );
}

const END_TEXT: Record<EndReason, string> = {
  player_canceled: 'You ended the session.',
  player_left:
    'This page was closed for two minutes, so your spot went to the next player.',
  player_disconnected: 'You left the Slippi session.',
  no_show:
    'HAL waited but no connection arrived. Queue again when you are ready.',
  idle_timeout: 'No game started for a while, so HAL freed the slot.',
  yielded:
    'Others were waiting, so HAL moved on after your game. Thanks for playing.',
  service_failure:
    'HAL had a problem it could not recover from. Queue again whenever you like.',
};

function endText(reason: EndReason | null): string {
  return reason === null ? 'The session ended.' : END_TEXT[reason];
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
    stage: null,
    desired_return:
      desired === undefined ? options.default_desired_return : desired,
  };
}
