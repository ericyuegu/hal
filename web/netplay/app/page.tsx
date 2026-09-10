'use client';

import { useCallback, useEffect, useState } from 'react';
import type { SyntheticEvent } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Gamepad2,
  LoaderCircle,
  RotateCcw,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  cancelJob,
  Capacity,
  createJob,
  CreateJob,
  fallbackOptions,
  getCapacity,
  getJob,
  getOptions,
  Job,
  Options,
  requestRematch,
} from '@/lib/netplay-api';

type SavedJob = { id: string; token: string };
const savedJobKey = 'hal-netplay-job-v1';
const terminal = new Set(['complete', 'failed', 'canceled', 'no_show']);
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

function readSavedJob(): SavedJob | null {
  if (typeof window === 'undefined') return null;
  const raw = localStorage.getItem(savedJobKey);
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<SavedJob>;
    if (typeof value.id === 'string' && typeof value.token === 'string') {
      return { id: value.id, token: value.token };
    }
  } catch {
    // Remove invalid device-local credentials below.
  }
  localStorage.removeItem(savedJobKey);
  return null;
}

export default function Home() {
  const [options, setOptions] = useState<Options>(fallbackOptions);
  const [capacity, setCapacity] = useState<Capacity>(unavailableCapacity);
  const [saved, setSaved] = useState<SavedJob | null>(readSavedJob);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const remember = useCallback((credentials: SavedJob) => {
    localStorage.setItem(savedJobKey, JSON.stringify(credentials));
    setSaved(credentials);
  }, []);

  const forget = useCallback(() => {
    localStorage.removeItem(savedJobKey);
    setSaved(null);
    setJob(null);
    setError('');
  }, []);

  const join = useCallback(
    async (values: CreateJob) => {
      setBusy(true);
      setError('');
      try {
        const created = await createJob(values);
        remember({ id: created.id, token: created.token });
        setJob(created);
        return created;
      } catch (cause) {
        const message =
          cause instanceof Error ? cause.message : 'Could not join the queue.';
        setError(message);
        throw cause;
      } finally {
        setBusy(false);
      }
    },
    [remember],
  );

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
        if (!canceled)
          setError(
            cause instanceof Error
              ? cause.message
              : 'Could not read reservation status.',
          );
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => {
      canceled = true;
      window.clearInterval(timer);
    };
  }, [saved]);

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
            online_delay: { type: 'integer', enum: [2, 3] },
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
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not cancel the reservation.',
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen px-5 py-6 sm:px-8 lg:px-12 lg:py-10">
      <div className="mx-auto max-w-[1180px]">
        <Header capacity={capacity} />
        <div className="grid gap-8 pt-9 lg:grid-cols-[minmax(0,1fr)_330px] lg:gap-12">
          <section aria-labelledby="queue-title">
            {job ? (
              <Reservation
                job={job}
                options={options}
                saved={saved!}
                busy={busy}
                error={error}
                cancel={cancel}
                forget={forget}
                update={setJob}
                setBusy={setBusy}
                setError={setError}
              />
            ) : (
              <JoinForm
                options={options}
                capacity={capacity}
                busy={busy}
                error={error}
                join={join}
              />
            )}
          </section>
          <QueueAside capacity={capacity} />
        </div>
      </div>
    </main>
  );
}

function Header({ capacity }: { capacity: Capacity }) {
  const status = serviceStatus(capacity.service_status);
  return (
    <header className="flex items-center justify-between border-b border-white/10 pb-5">
      <div className="flex items-center gap-3">
        <div className="logo-mark">H</div>
        <div>
          <p className="text-base font-semibold tracking-[0.18em]">HAL</p>
          <p className="text-xs text-muted-foreground">Direct netplay</p>
        </div>
      </div>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span
          className={`status-pulse ${capacity.service_status}`}
          aria-hidden="true"
        />
        {status}
      </div>
    </header>
  );
}

function JoinForm({
  options,
  capacity,
  busy,
  error,
  join,
}: {
  options: Options;
  capacity: Capacity;
  busy: boolean;
  error: string;
  join: (values: CreateJob) => Promise<Job>;
}) {
  const [character, setCharacter] = useState('FOX');
  const [imitation, setImitation] = useState('IBDW#0');
  const [delay, setDelay] = useState('2');

  function submit(event: SyntheticEvent<HTMLFormElement, SubmitEvent>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const playerCode = form.get('player_code');
    if (typeof playerCode !== 'string') return;
    void join({
      player_code: playerCode.toUpperCase(),
      character,
      imitation,
      online_delay: Number(delay),
    }).catch(() => undefined);
  }

  return (
    <>
      <div className="mb-7 max-w-2xl">
        <p className="eyebrow">PLAY THE MODEL</p>
        <h1
          id="queue-title"
          className="mt-3 text-3xl font-semibold tracking-[-0.035em] sm:text-4xl"
        >
          Set up your match
        </h1>
        <p className="mt-3 max-w-xl text-base leading-7 text-muted-foreground">
          Choose how HAL plays. When a slot opens, connect through Slippi and
          start the set.
        </p>
      </div>
      <form onSubmit={submit} className="surface p-5 sm:p-7">
        <div className="grid gap-6 sm:grid-cols-2">
          <Field
            label="Your player code"
            htmlFor="player-code"
            hint="Use the exact code shown in Slippi."
          >
            <Input
              id="player-code"
              name="player_code"
              placeholder="CRYO#610"
              autoCapitalize="characters"
              required
            />
          </Field>
          <ChoiceSelect
            label="HAL character"
            id="character"
            value={character}
            choices={options.characters}
            set={setCharacter}
          />
          <ChoiceSelect
            label="Play like"
            id="imitation"
            value={imitation}
            choices={options.imitations}
            set={setImitation}
            hint="This changes policy conditioning, not matchmaking."
          />
        </div>
        <fieldset className="mt-7 border-t border-white/10 pt-6">
          <legend className="text-sm font-medium">Frame delay</legend>
          <RadioGroup
            value={delay}
            onValueChange={(value) => setDelay(value ?? '2')}
            className="mt-3 grid gap-3 sm:grid-cols-2"
          >
            <DelayChoice
              value="2"
              title="2 frames"
              description="Lower latency · recommended"
            />
            <DelayChoice
              value="3"
              title="3 frames"
              description="More network tolerance"
            />
          </RadioGroup>
        </fieldset>
        <div className="mt-7 flex flex-col gap-4 border-t border-white/10 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <CircleDot className="size-4 text-cyan-300" /> Game 1 starts on a
            random legal stage.
          </p>
          <Button
            type="submit"
            size="lg"
            className="queue-button"
            disabled={busy || capacity.healthy_slots === 0}
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
        {capacity.healthy_slots === 0 && (
          <output className="mt-4 block text-sm text-amber-200">
            {capacity.service_message}
          </output>
        )}
        <ErrorMessage value={error} />
      </form>
    </>
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
  update: (job: Job) => void;
  setBusy: (value: boolean) => void;
  setError: (value: string) => void;
}) {
  const ended = terminal.has(job.status);
  return (
    <>
      <div className="mb-7">
        <p className="eyebrow">YOUR RESERVATION</p>
        <h1
          id="queue-title"
          className="mt-3 text-3xl font-semibold tracking-[-0.035em] sm:text-4xl"
        >
          {statusTitle(job)}
        </h1>
        <p className="mt-3 text-base text-muted-foreground">
          {job.player_code} · {label(options.characters, job.character)} · delay{' '}
          {job.online_delay}
        </p>
      </div>
      <div className="surface overflow-hidden">
        <div className="status-panel p-6 sm:p-8">
          <StatusBody job={job} />
        </div>
        {job.status === 'rematch_wait' && (
          <RematchForm
            job={job}
            saved={saved}
            options={options}
            busy={busy}
            update={update}
            setBusy={setBusy}
            setError={setError}
          />
        )}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-6 py-5 sm:px-8">
          <p className="text-sm text-muted-foreground">
            Game {Math.min(job.game_count + 1, 5)} of up to 5
          </p>
          {ended ? (
            <Button onClick={forget} variant="secondary">
              <RotateCcw /> New reservation
            </Button>
          ) : (
            <Button
              onClick={() => void cancel()}
              variant="ghost"
              disabled={busy || job.cancel_after_game}
            >
              <X />{' '}
              {job.status === 'playing'
                ? 'Stop after this game'
                : 'Leave queue'}
            </Button>
          )}
        </div>
      </div>
      <ErrorMessage value={error} />
    </>
  );
}

function StatusBody({ job }: { job: Job }) {
  if (job.status === 'queued') {
    return (
      <BigStatus
        value={String(job.queue_position ?? '—')}
        label="Your place in queue"
        detail="Keep this page open."
      />
    );
  }
  if (job.status === 'leased') {
    return (
      <BigStatus
        icon={<LoaderCircle className="size-8 animate-spin" />}
        label="Preparing your Dolphin"
        detail="Your slot is reserved."
      />
    );
  }
  if (job.status === 'connecting') {
    return (
      <BigStatus
        value={job.connect_code ?? '—'}
        label="Direct-connect to HAL now"
        detail={
          <Countdown deadline={job.connect_deadline} suffix=" to connect" />
        }
      />
    );
  }
  if (job.status === 'playing') {
    return (
      <BigStatus
        icon={<Gamepad2 className="size-9" />}
        label="Match in progress"
        detail={
          job.cancel_after_game
            ? 'This reservation will end after the game.'
            : 'The page will update when the game ends.'
        }
      />
    );
  }
  if (job.status === 'rematch_wait') {
    return (
      <BigStatus
        value={job.last_result ?? 'Game complete'}
        label={
          job.actual_stage
            ? `Played on ${pretty(job.actual_stage)}`
            : 'Game complete'
        }
        detail={
          <Countdown
            deadline={job.rematch_deadline}
            suffix=" to choose a rematch"
          />
        }
      />
    );
  }
  if (job.status === 'rematch_ready') {
    return (
      <BigStatus
        icon={<LoaderCircle className="size-8 animate-spin" />}
        label="Setting up the rematch"
        detail="Keep the Slippi connection open."
      />
    );
  }
  return (
    <BigStatus
      icon={<CheckCircle2 className="size-9" />}
      label={statusTitle(job)}
      detail={
        job.error_code
          ? `Reason: ${pretty(job.error_code)}`
          : 'You can start a new reservation.'
      }
    />
  );
}

function RematchForm({
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
  const [character, setCharacter] = useState(job.character);
  const [imitation, setImitation] = useState(job.imitation);
  const [stage, setStage] = useState(job.requested_stage ?? 'BATTLEFIELD');

  async function submit(event: SyntheticEvent<HTMLFormElement, SubmitEvent>) {
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
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not request the rematch.',
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={submit}
      className="grid gap-4 border-t border-white/10 bg-black/10 p-6 sm:grid-cols-3 sm:p-8"
    >
      <ChoiceSelect
        label="Character"
        id="rematch-character"
        value={character}
        choices={options.characters}
        set={setCharacter}
      />
      <ChoiceSelect
        label="Play like"
        id="rematch-imitation"
        value={imitation}
        choices={options.imitations}
        set={setImitation}
      />
      <ChoiceSelect
        label="Requested stage"
        id="rematch-stage"
        value={stage}
        choices={options.stages}
        set={setStage}
      />
      <p className="text-xs leading-5 text-muted-foreground sm:col-span-2">
        HAL selects this stage only if it lost. If you lost, choose the stage in
        Slippi.
      </p>
      <Button type="submit" disabled={busy} className="sm:justify-self-end">
        {busy ? <LoaderCircle className="animate-spin" /> : 'Play again'}
      </Button>
    </form>
  );
}

function QueueAside({ capacity }: { capacity: Capacity }) {
  const fps = capacity.game_fps;
  return (
    <aside className="space-y-5" aria-label="Queue information">
      <section className="surface overflow-hidden">
        <div className="border-b border-white/10 px-5 py-4">
          <p className="eyebrow">LIVE CAPACITY</p>
        </div>
        <div className="grid grid-cols-2 divide-x divide-white/10">
          <Stat
            value={`${capacity.healthy_slots}/${capacity.capacity}`}
            label="Healthy slots"
          />
          <Stat value={String(capacity.queued)} label="In queue" />
        </div>
        <div className="border-t border-white/10 px-5 py-4">
          <p className="text-sm font-medium">
            {serviceStatus(capacity.service_status)}
            {fps === null ? '' : ` · ${fps.toFixed(1)} FPS`}
          </p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            {capacity.service_message}
          </p>
        </div>
      </section>
      <section className="surface p-5">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Gamepad2 className="size-4 text-violet-300" /> After you join
        </h2>
        <ol className="steps mt-5 space-y-5 text-sm">
          <li>Keep this page open while you wait.</li>
          <li>When assigned, direct-connect to HAL in Slippi.</li>
          <li>Play up to five games on the same connection.</li>
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
  choices: { value: string; label: string }[];
  set: (value: string) => void;
  hint?: string;
}) {
  return (
    <Field label={title} htmlFor={id} hint={hint}>
      <Select value={value} onValueChange={(next) => set(next ?? value)}>
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

function DelayChoice({
  value,
  title,
  description,
}: {
  value: string;
  title: string;
  description: string;
}) {
  return (
    <Label className="choice-card">
      <RadioGroupItem value={value} />
      <span>
        <span className="block font-medium">{title}</span>
        <span className="mt-1 block text-sm font-normal text-muted-foreground">
          {description}
        </span>
      </span>
    </Label>
  );
}

function BigStatus({
  value,
  icon,
  label: title,
  detail,
}: {
  value?: string;
  icon?: React.ReactNode;
  label: string;
  detail: React.ReactNode;
}) {
  return (
    <div>
      <div className="status-value">{value ?? icon}</div>
      <h2 className="mt-4 text-lg font-semibold">{title}</h2>
      <div className="mt-2 text-sm text-muted-foreground">{detail}</div>
    </div>
  );
}

function Countdown({
  deadline,
  suffix,
}: {
  deadline: number | null;
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
  return (
    <>
      {seconds === null ? '—' : `${seconds}s`}
      {suffix}
    </>
  );
}

function Field({
  label: title,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <Label htmlFor={htmlFor} className="mb-2 text-sm">
        {title}
      </Label>
      {children}
      {hint && (
        <p className="mt-2 text-xs leading-5 text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}

function Stat({ value, label: title }: { value: string; label: string }) {
  return (
    <div className="px-5 py-5">
      <p className="font-mono text-2xl font-semibold text-cyan-200">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{title}</p>
    </div>
  );
}

function ErrorMessage({ value }: { value: string }) {
  return value ? (
    <p
      role="alert"
      className="mt-4 rounded-lg border border-red-300/20 bg-red-300/5 px-4 py-3 text-sm text-red-100"
    >
      {value}
    </p>
  ) : null;
}

function statusTitle(job: Job): string {
  return (
    {
      queued: 'Waiting for a slot',
      leased: 'Starting HAL',
      connecting: 'Your slot is ready',
      playing: 'Game in progress',
      rematch_wait: 'Run it back?',
      rematch_ready: 'Preparing rematch',
      complete: 'Set complete',
      failed: 'Service error',
      canceled: 'Reservation canceled',
      no_show: 'Connection timed out',
    }[job.status] ?? pretty(job.status)
  );
}

function label(
  choices: { value: string; label: string }[],
  value: string,
): string {
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
  if (values.online_delay !== 2 && values.online_delay !== 3)
    throw new Error('online_delay must be 2 or 3.');
  return values as CreateJob;
}
