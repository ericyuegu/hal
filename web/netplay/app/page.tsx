'use client';

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import type { ReactNode, SyntheticEvent } from 'react';
import {
  ArrowRight,
  Check,
  ChevronDown,
  Copy,
  Gamepad2,
  LoaderCircle,
  RotateCcw,
  Settings2,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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

type SavedJob = { id: string; token: string };
type Prefs = {
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
};

const savedJobKey = 'hal-netplay-job-v1';
const prefsKey = 'hal-netplay-prefs-v1';
const terminal = new Set(['complete', 'failed', 'canceled', 'no_show']);
// Mirrors hal/netplay_service/domain.py validate_player_code.
const playerCodePattern = /^[A-Z0-9]{1,8}#[0-9]{1,4}$/;
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
    (prefs.desired_return === null ||
      typeof prefs.desired_return === 'number') &&
    typeof prefs.temperature === 'number'
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
        desired_return: values.desired_return ?? null,
        temperature: values.temperature ?? 1,
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
              type: ['number', 'null'],
              minimum: 0,
              maximum: 40,
            },
            temperature: { type: 'number', minimum: 0.8, maximum: 1.1 },
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
    body = <LoadingCard />;
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
    <main className="shell">
      <Header capacity={capacity} />
      <div className="layout">
        <section aria-labelledby="page-title" className="min-w-0">
          {body}
        </section>
        <QueueAside capacity={capacity} options={options} />
      </div>
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
    document.title = job ? `${tabTitle(job)} · HAL` : 'HAL Netplay';
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

function Header({ capacity }: { capacity: Capacity | null }) {
  const status = capacity?.service_status ?? 'loading';
  return (
    <header className="topbar">
      <div className="flex items-center gap-3">
        <div className="logo-mark" aria-hidden="true">
          <span />
        </div>
        <div className="leading-tight">
          <p className="wordmark">HAL</p>
          <p className="text-xs text-muted-foreground">Slippi direct netplay</p>
        </div>
      </div>
      <div className="pill" data-status={status}>
        <span className="dot" aria-hidden="true" />
        {capacity ? serviceStatus(capacity.service_status) : 'Connecting…'}
      </div>
    </header>
  );
}

function PageTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-6">
      <p className="eyebrow">{eyebrow}</p>
      <h1 id="page-title" className="page-title">
        {title}
      </h1>
      {children}
    </div>
  );
}

function LoadingCard() {
  return (
    <>
      <PageTitle eyebrow="Loading" title="One moment" />
      <div className="card grid min-h-72 place-items-center">
        <LoaderCircle className="size-6 animate-spin text-muted-foreground" />
      </div>
    </>
  );
}

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
  const [character, setCharacter] = useState(prefs?.character ?? 'FOX');
  const [imitation, setImitation] = useState(prefs?.imitation ?? 'IBDW#0');
  const [delay, setDelay] = useState(prefs?.online_delay ?? 2);
  const [returnTarget, setReturnTarget] = useState<number | null>(
    prefs ? prefs.desired_return : fallbackOptions.default_desired_return,
  );
  const [temperature, setTemperature] = useState(
    prefs?.temperature ?? fallbackOptions.default_temperature,
  );

  // Stored choices can outlive what the current policy offers.
  const characterValue = has(options.characters, character) ? character : 'FOX';
  const imitationValue = has(options.imitations, imitation)
    ? imitation
    : options.imitations[0].value;
  const delayValue = options.online_delays.includes(delay)
    ? delay
    : options.online_delays[0];

  const codeValid = playerCodePattern.test(playerCode);
  const showCodeError = touched && playerCode !== '' && !codeValid;
  const unavailable = capacity !== null && capacity.healthy_slots === 0;

  function submit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched(true);
    if (!codeValid) return;
    void join({
      player_code: playerCode,
      character: characterValue,
      imitation: imitationValue,
      online_delay: delayValue,
      desired_return: returnTarget,
      temperature,
    }).catch(() => undefined);
  }

  return (
    <>
      <PageTitle eyebrow="Play the model" title="Set up your match">
        <p className="lede">
          Pick how HAL plays and join the queue. When a slot opens,
          direct-connect in Slippi and play up to {options.max_games} games.
        </p>
      </PageTitle>
      <form onSubmit={submit} className="card" noValidate>
        <div className="card-section grid gap-5 sm:grid-cols-[1fr_1fr]">
          <Field
            label="Your connect code"
            htmlFor="player-code"
            hint={
              showCodeError
                ? 'Use the format CODE#123, exactly as Slippi shows it.'
                : 'Exactly as shown in Slippi. Remembered on this device.'
            }
            invalid={showCodeError}
          >
            <Input
              id="player-code"
              name="player_code"
              className="control code-input"
              placeholder="CODE#123"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              maxLength={13}
              value={playerCode}
              aria-invalid={showCodeError || undefined}
              onBlur={() => setTouched(true)}
              onChange={(event) =>
                setPlayerCode(
                  event.target.value
                    .toUpperCase()
                    .replaceAll('＃', '#')
                    .replace(/\s/g, ''),
                )
              }
              required
            />
          </Field>
          <DelayField options={options} value={delayValue} set={setDelay} />
        </div>
        <div className="card-section grid gap-5 sm:grid-cols-2">
          <ChoiceSelect
            label="HAL plays"
            id="character"
            value={characterValue}
            choices={options.characters}
            set={setCharacter}
          />
          <ChoiceSelect
            label="In the style of"
            id="imitation"
            value={imitationValue}
            choices={options.imitations}
            set={setImitation}
            hint="Conditions the policy on a player; it does not affect matchmaking."
          />
        </div>
        <details className="card-section advanced">
          <summary>
            <Settings2 className="size-4 text-muted-foreground" />
            <span className="font-medium">Policy tuning</span>
            <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
              {returnTarget === null
                ? 'Unconditioned'
                : `Return ${returnTarget}`}{' '}
              · T {temperature.toFixed(2)}
            </span>
            <ChevronDown className="chevron size-4 text-muted-foreground" />
          </summary>
          <div className="pt-5">
            <PolicyFields
              options={options}
              returnTarget={returnTarget}
              setReturnTarget={setReturnTarget}
              temperature={temperature}
              setTemperature={setTemperature}
            />
            <p className="mt-4 text-xs text-muted-foreground">
              You can also change these while a game is running.
            </p>
          </div>
        </details>
        <div className="card-footer">
          <p className="text-sm text-muted-foreground">
            Game 1 is on a random legal stage.
          </p>
          <Button
            type="submit"
            size="lg"
            className="cta"
            disabled={busy || unavailable}
          >
            {busy ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <>
                Join queue <ArrowRight />
              </>
            )}
          </Button>
        </div>
        {unavailable && (
          <output className="notice mx-5 mb-5 sm:mx-7">
            {capacity.service_message}
          </output>
        )}
      </form>
      <ErrorMessage value={error} dismiss={dismissError} />
    </>
  );
}

function DelayField({
  options,
  value,
  set,
}: {
  options: Options;
  value: number;
  set: (value: number) => void;
}) {
  return (
    <fieldset>
      <legend className="field-label">Frame delay</legend>
      <div className="segmented">
        {options.online_delays.map((frames) => (
          <label key={frames}>
            <input
              type="radio"
              name="online_delay"
              className="sr-only"
              checked={frames === value}
              onChange={() => set(frames)}
            />
            {frames} frames
          </label>
        ))}
      </div>
      <p className="field-hint">
        {value === Math.min(...options.online_delays)
          ? 'Lowest latency. Recommended.'
          : 'More tolerant of a weak connection.'}
      </p>
    </fieldset>
  );
}

function PolicyFields({
  options,
  returnTarget,
  setReturnTarget,
  temperature,
  setTemperature,
}: {
  options: Options;
  returnTarget: number | null;
  setReturnTarget: (value: number | null) => void;
  temperature: number;
  setTemperature: (value: number) => void;
}) {
  const [returnMin, returnMax] = options.desired_return_range;
  const [tempMin, tempMax] = options.temperature_range;
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div>
        <div className="flex items-baseline justify-between">
          <label htmlFor="return-target" className="field-label">
            Return target
          </label>
          <span className="readout">
            {returnTarget === null ? 'Off' : returnTarget}
          </span>
        </div>
        <input
          id="return-target"
          type="range"
          className="range"
          min={returnMin}
          max={returnMax}
          step={1}
          disabled={returnTarget === null}
          value={returnTarget ?? options.default_desired_return}
          onChange={(event) => setReturnTarget(Number(event.target.value))}
        />
        <div className="mt-2 flex items-center justify-between gap-3">
          <label className="toggle">
            <input
              type="checkbox"
              checked={returnTarget === null}
              onChange={(event) =>
                setReturnTarget(
                  event.target.checked ? null : options.default_desired_return,
                )
              }
            />
            Unconditioned
          </label>
          <ResetLink
            visible={returnTarget !== options.default_desired_return}
            reset={() => setReturnTarget(options.default_desired_return)}
          />
        </div>
        <p className="field-hint">
          Higher asks for stronger play. {options.default_desired_return} is
          about the top 10% of training games.
        </p>
      </div>
      <div>
        <div className="flex items-baseline justify-between">
          <label htmlFor="temperature" className="field-label">
            Temperature
          </label>
          <span className="readout">{temperature.toFixed(2)}</span>
        </div>
        <input
          id="temperature"
          type="range"
          className="range"
          min={tempMin}
          max={tempMax}
          step={0.01}
          value={temperature}
          onChange={(event) => setTemperature(Number(event.target.value))}
        />
        <div className="mt-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
          <span>Focused</span>
          <ResetLink
            visible={temperature !== options.default_temperature}
            reset={() => setTemperature(options.default_temperature)}
          />
          <span>Varied</span>
        </div>
        <p className="field-hint">
          Higher values add variety to HAL&apos;s play.
        </p>
      </div>
    </div>
  );
}

function ResetLink({
  visible,
  reset,
}: {
  visible: boolean;
  reset: () => void;
}) {
  return (
    <button
      type="button"
      className="reset-link"
      onClick={reset}
      data-visible={visible}
      tabIndex={visible ? 0 : -1}
      aria-hidden={!visible}
    >
      Reset
    </button>
  );
}

function PolicyForm({
  job,
  saved,
  options,
  busy,
  update,
  setBusy,
  setError,
}: {
  job: Job;
  saved: SavedJob;
  options: Options;
  busy: boolean;
  update: (job: Job) => void;
  setBusy: (value: boolean) => void;
  setError: (value: string) => void;
}) {
  const [returnTarget, setReturnTarget] = useState<number | null>(
    job.desired_return,
  );
  const [temperature, setTemperature] = useState(job.temperature);
  const [applied, setApplied] = useState(false);
  const dirty =
    returnTarget !== job.desired_return || temperature !== job.temperature;

  useEffect(() => {
    if (!applied) return;
    const timer = window.setTimeout(() => setApplied(false), 2000);
    return () => window.clearTimeout(timer);
  }, [applied]);

  async function submit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      update(
        await updatePolicy(saved.id, saved.token, {
          desired_return: returnTarget,
          temperature,
        }),
      );
      setApplied(true);
    } catch (cause) {
      setError(errorText(cause, 'Could not update policy settings.'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="card-section advanced">
      <summary>
        <Settings2 className="size-4 text-muted-foreground" />
        <span className="font-medium">Policy tuning</span>
        <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
          {job.desired_return === null
            ? 'Unconditioned'
            : `Return ${job.desired_return}`}{' '}
          · T {job.temperature.toFixed(2)}
        </span>
        <ChevronDown className="chevron size-4 text-muted-foreground" />
      </summary>
      <form onSubmit={(event) => void submit(event)} className="pt-5">
        <PolicyFields
          options={options}
          returnTarget={returnTarget}
          setReturnTarget={setReturnTarget}
          temperature={temperature}
          setTemperature={setTemperature}
        />
        <div className="mt-5 flex items-center gap-3">
          <Button type="submit" variant="secondary" disabled={busy || !dirty}>
            Apply
          </Button>
          <span className="text-xs text-muted-foreground" aria-live="polite">
            {applied ? (
              <span className="inline-flex items-center gap-1 text-[var(--good)]">
                <Check className="size-3.5" /> Applied at the next replan
              </span>
            ) : dirty ? (
              'Unsaved changes'
            ) : (
              'Takes effect mid-game'
            )}
          </span>
        </div>
      </form>
    </details>
  );
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
  return (
    <>
      <PageTitle eyebrow="Your reservation" title={statusTitle(job)}>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="tag font-mono">{job.player_code}</span>
          <span className="tag">
            vs {label(options.characters, job.character)}
          </span>
          <span className="tag">
            {label(options.imitations, job.imitation)}
          </span>
          <span className="tag">{job.online_delay}f delay</span>
        </div>
      </PageTitle>
      <div className="card overflow-hidden">
        <Progress job={job} options={options} />
        <div className="status-panel" data-status={job.status}>
          <StatusBody job={job} options={options} />
        </div>
        {job.status === 'rematch_wait' && (
          <RematchForm
            key={job.game_count}
            job={job}
            saved={saved}
            options={options}
            busy={busy}
            cancel={cancel}
            update={update}
            setBusy={setBusy}
            setError={setError}
          />
        )}
        {!ended && (
          <PolicyForm
            key={job.id}
            job={job}
            saved={saved}
            options={options}
            busy={busy}
            update={update}
            setBusy={setBusy}
            setError={setError}
          />
        )}
        <div className="card-footer">
          <p className="text-sm text-muted-foreground tabular-nums">
            {ended
              ? `${job.game_count} of ${options.max_games} games played`
              : `Game ${Math.min(job.game_count + 1, options.max_games)} of up to ${options.max_games}`}
          </p>
          {ended ? (
            <div className="flex flex-wrap gap-2">
              <Button onClick={forget} variant="ghost">
                Change settings
              </Button>
              <Button onClick={requeue} className="cta" disabled={busy}>
                <RotateCcw /> Queue again
              </Button>
            </div>
          ) : (
            job.status !== 'rematch_wait' && (
              <Button
                onClick={() => void cancel()}
                variant="ghost"
                disabled={busy || job.cancel_after_game}
              >
                <X /> {cancelLabel(job)}
              </Button>
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
          <span className="progress-dot" aria-hidden="true">
            {index < current ? <Check className="size-3" /> : index + 1}
          </span>
          <span className="progress-label">
            {step.label === 'Play' && job.game_count > 0
              ? `Game ${Math.min(job.game_count + (job.status === 'rematch_wait' ? 0 : 1), options.max_games)}`
              : step.label}
          </span>
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
        label={job.queue_position === 1 ? "You're next" : 'Place in queue'}
        detail="Keep this tab open. We'll notify you when your slot is ready."
      />
    );
  }
  if (job.status === 'leased') {
    return (
      <BigStatus
        icon={<LoaderCircle className="size-10 animate-spin" />}
        label="Booting HAL's Dolphin"
        detail="Your slot is reserved. Open Slippi so you're ready to connect."
      />
    );
  }
  if (job.status === 'connecting') {
    return (
      <div className="w-full">
        <p className="field-label">Direct-connect to</p>
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
        icon={<Gamepad2 className="size-10" />}
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
      <div className="w-full">
        <BigStatus
          value={resultText(job.last_result)}
          label={
            job.actual_stage
              ? `Game ${job.game_count} · ${label(options.stages, job.actual_stage)}`
              : `Game ${job.game_count} complete`
          }
          detail="Stay on the Slippi results screen while you choose."
        />
        <div className="mt-6">
          <Countdown
            deadline={job.rematch_deadline}
            total={options.rematch_seconds}
            suffix="to start the next game"
          />
        </div>
      </div>
    );
  }
  if (job.status === 'rematch_ready') {
    return (
      <BigStatus
        icon={<LoaderCircle className="size-10 animate-spin" />}
        label="Setting up the rematch"
        detail="Keep the Slippi connection open."
      />
    );
  }
  return (
    <BigStatus
      icon={
        job.status === 'complete' ? (
          <Check className="size-10" />
        ) : (
          <X className="size-10" />
        )
      }
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
            <Check className="size-4" /> Copied
          </>
        ) : (
          <>
            <Copy className="size-4" /> Copy
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
  cancel,
  update,
  setBusy,
  setError,
}: {
  job: Job;
  saved: SavedJob;
  options: Options;
  busy: boolean;
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

  async function submit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
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
    <form
      onSubmit={(event) => void submit(event)}
      className="card-section rematch"
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <ChoiceSelect
          label="HAL plays"
          id="rematch-character"
          value={character}
          choices={options.characters}
          set={setCharacter}
        />
        <ChoiceSelect
          label="In the style of"
          id="rematch-imitation"
          value={imitation}
          choices={options.imitations}
          set={setImitation}
        />
        <ChoiceSelect
          label="HAL's stage pick"
          id="rematch-stage"
          value={stage}
          choices={options.stages}
          set={setStage}
        />
      </div>
      <p className="field-hint">
        HAL picks the stage only if it lost the last game. If you lost, pick it
        in Slippi.
      </p>
      <div className="mt-5 flex flex-wrap items-center justify-end gap-2">
        <Button
          type="button"
          variant="ghost"
          disabled={busy}
          onClick={() => void cancel()}
        >
          End set
        </Button>
        <Button type="submit" disabled={busy} className="cta">
          {busy ? (
            <LoaderCircle className="animate-spin" />
          ) : (
            <>
              Play game {job.game_count + 1} <ArrowRight />
            </>
          )}
        </Button>
      </div>
    </form>
  );
}

function QueueAside({
  capacity,
  options,
}: {
  capacity: Capacity | null;
  options: Options;
}) {
  const open = capacity
    ? Math.max(capacity.healthy_slots - capacity.active, 0)
    : null;
  return (
    <aside className="aside" aria-label="Service information">
      <section className="card">
        <div className="stats">
          <Stat value={open === null ? '–' : String(open)} label="Open slots" />
          <Stat
            value={capacity ? String(capacity.queued) : '–'}
            label="Waiting"
          />
          <Stat
            value={
              capacity?.game_fps == null ? '–' : capacity.game_fps.toFixed(0)
            }
            label="Game FPS"
          />
        </div>
        <p className="border-t border-[var(--hairline)] px-5 py-4 text-xs leading-5 text-muted-foreground">
          {capacity?.service_message ?? 'Checking service status…'}
        </p>
      </section>
      <section className="card p-5">
        <h2 className="text-sm font-semibold">How it works</h2>
        <ol className="steps mt-4 space-y-4 text-sm">
          <li>Join the queue and keep this tab open.</li>
          <li>
            When your slot is ready, direct-connect in Slippi. You have{' '}
            {minutes(options.no_show_seconds)}.
          </li>
          <li>
            Play up to {options.max_games} games. Between games you have{' '}
            {minutes(options.rematch_seconds)} to call a rematch.
          </li>
        </ol>
      </section>
    </aside>
  );
}

function serviceStatus(status: Capacity['service_status']): string {
  return status[0].toUpperCase() + status.slice(1);
}

function ChoiceSelect({
  label: title,
  id,
  value,
  choices,
  set,
  hint,
}: {
  label: string;
  id: string;
  value: string;
  choices: Choice[];
  set: (value: string) => void;
  hint?: string;
}) {
  return (
    <Field label={title} htmlFor={id} hint={hint}>
      <Select
        items={choices}
        value={value}
        onValueChange={(next) => set(next ?? value)}
      >
        <SelectTrigger id={id} className="control w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {choices.map((choice) => (
            <SelectItem key={choice.value} value={choice.value}>
              {choice.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
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
      <h2 className="mt-3 text-lg font-semibold tracking-tight">{title}</h2>
      <p className="mt-1.5 max-w-md text-sm text-muted-foreground">{detail}</p>
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
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-mono text-base font-semibold tabular-nums">
          {seconds === null ? '–:––' : clock(seconds)}
        </span>
        <span className="text-muted-foreground">{suffix}</span>
      </div>
      <div className="meter" aria-hidden="true">
        <span style={{ transform: `scaleX(${fraction})` }} />
      </div>
    </div>
  );
}

function Field({
  label: title,
  htmlFor,
  hint,
  invalid,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  invalid?: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      <label htmlFor={htmlFor} className="field-label">
        {title}
      </label>
      {children}
      {hint && (
        <p className="field-hint" data-invalid={invalid || undefined}>
          {hint}
        </p>
      )}
    </div>
  );
}

function Stat({ value, label: title }: { value: string; label: string }) {
  return (
    <div className="stat">
      <p className="stat-value">{value}</p>
      <p className="stat-label">{title}</p>
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
        <X className="size-4" />
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
  if (!options.characters.some((choice) => choice.value === values.character))
    throw new Error('character is not available.');
  if (!options.imitations.some((choice) => choice.value === values.imitation))
    throw new Error('imitation is not available.');
  if (!options.online_delays.includes(values.online_delay as number))
    throw new Error('online_delay is not available.');
  if (
    values.desired_return !== undefined &&
    values.desired_return !== null &&
    (typeof values.desired_return !== 'number' ||
      !Number.isFinite(values.desired_return) ||
      values.desired_return < options.desired_return_range[0] ||
      values.desired_return > options.desired_return_range[1])
  )
    throw new Error('desired_return is out of range.');
  if (
    values.temperature !== undefined &&
    (typeof values.temperature !== 'number' ||
      !Number.isFinite(values.temperature) ||
      values.temperature < options.temperature_range[0] ||
      values.temperature > options.temperature_range[1])
  )
    throw new Error('temperature is out of range.');
  return values as CreateJob;
}
