export type Choice = { value: string; label: string };

export type Options = {
  characters: Choice[];
  imitations: Choice[];
  stages: Choice[];
  online_delays: number[];
  desired_return_range: [number, number];
  default_desired_return: number;
  temperature_range: [number, number];
  default_temperature: number;
};

export type Capacity = {
  capacity: number;
  healthy_slots: number;
  active: number;
  queued: number;
  service_status: 'ready' | 'degraded' | 'recovering' | 'unavailable';
  service_message: string;
  target_fps: number;
  game_fps: number | null;
  frame_interval_p95_ms: number | null;
  dolphin_step_p95_ms: number | null;
  policy_round_trip_p95_ms: number | null;
  model_inference_p95_ms: number | null;
  batch_wait_p95_ms: number | null;
  recoveries: number;
};

export type Phase =
  | 'booting'
  | 'waiting_for_player'
  | 'character_select'
  | 'in_game'
  | 'paused';

export type EndReason =
  | 'player_canceled'
  | 'player_left'
  | 'player_disconnected'
  | 'no_show'
  | 'idle_timeout'
  | 'yielded'
  | 'service_failure';

export type Settings = {
  revision: number;
  character: string;
  imitation: string;
  stage: string | null;
  desired_return: number | null;
  temperature: number;
};

export type FinishedGame = {
  number: number;
  stage: string;
  result: 'win' | 'loss' | 'no_contest';
};

export type Job = {
  id: string;
  player_code: string;
  online_delay: number;
  status: 'queued' | 'assigned' | 'ended';
  end_reason: EndReason | null;
  queue_position: number | null;
  attempt: number;
  settings: Settings;
  observed: {
    seq: number;
    phase: Phase;
    bot_code: string | null;
    seen_revision: number;
    locked_revision: number | null;
  } | null;
  phase_deadline: number | null;
  games: FinishedGame[];
  wind_down: 'player' | 'yield' | null;
  lock_requests: number;
};

export type JobCredentials = Job & { token: string };

export type CreateJob = {
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  stage?: string | null;
  desired_return?: number | null;
  temperature?: number;
};

export type SettingsUpdate = Partial<Omit<Settings, 'revision'>>;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('Content-Type', 'application/json');
  const response = await fetch(path, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new ApiError(
      body?.detail ?? `Request failed (${response.status})`,
      response.status,
    );
  }
  return (await response.json()) as T;
}

function authorized(token: string, init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${token}`);
  return {
    ...init,
    headers,
  };
}

export function getOptions(): Promise<Options> {
  return request('/v1/options');
}

export function getCapacity(): Promise<Capacity> {
  return request('/v1/capacity');
}

export function createJob(values: CreateJob): Promise<JobCredentials> {
  return request('/v1/jobs', { method: 'POST', body: JSON.stringify(values) });
}

export function getJob(id: string, token: string): Promise<Job> {
  return request(`/v1/jobs/${id}`, authorized(token));
}

export function leaveJob(id: string, token: string): Promise<Job> {
  return request(`/v1/jobs/${id}`, authorized(token, { method: 'DELETE' }));
}

export function updateSettings(
  id: string,
  token: string,
  values: SettingsUpdate,
): Promise<Job> {
  return request(
    `/v1/jobs/${id}/settings`,
    authorized(token, { method: 'PATCH', body: JSON.stringify(values) }),
  );
}

export function requestLock(id: string, token: string): Promise<Job> {
  return request(`/v1/jobs/${id}/lock`, authorized(token, { method: 'POST' }));
}

export const fallbackOptions: Options = {
  characters: [
    ['FOX', 'Fox'],
    ['FALCO', 'Falco'],
    ['MARTH', 'Marth'],
    ['SHEIK', 'Sheik'],
    ['JIGGLYPUFF', 'Jigglypuff'],
    ['PEACH', 'Peach'],
    ['CPTFALCON', 'Captain Falcon'],
    ['POPO', 'Ice Climbers'],
    ['PIKACHU', 'Pikachu'],
    ['SAMUS', 'Samus'],
    ['YOSHI', 'Yoshi'],
    ['LUIGI', 'Luigi'],
    ['GANONDORF', 'Ganondorf'],
    ['MARIO', 'Mario'],
    ['DOC', 'Dr. Mario'],
    ['LINK', 'Link'],
    ['YOUNG_LINK', 'Young Link'],
    ['ZELDA', 'Zelda'],
    ['MEWTWO', 'Mewtwo'],
    ['NESS', 'Ness'],
    ['ROY', 'Roy'],
    ['GAMEANDWATCH', 'Mr. Game & Watch'],
    ['PICHU', 'Pichu'],
    ['DK', 'Donkey Kong'],
    ['BOWSER', 'Bowser'],
    ['KIRBY', 'Kirby'],
  ].map(([value, label]) => ({ value, label })),
  // The player roster comes from the published policy; before it loads, only ranks are offered.
  imitations: [
    ['PLATINUM', 'Platinum rank'],
    ['DIAMOND', 'Diamond rank'],
    ['MASTER', 'Master rank'],
    ['MASKED', 'No player identity'],
  ].map(([value, label]) => ({ value, label })),
  stages: [
    ['BATTLEFIELD', 'Battlefield'],
    ['FINAL_DESTINATION', 'Final Destination'],
    ['DREAMLAND', 'Dream Land'],
    ['POKEMON_STADIUM', 'Pokémon Stadium'],
    ['YOSHIS_STORY', "Yoshi's Story"],
    ['FOUNTAIN_OF_DREAMS', 'Fountain of Dreams'],
  ].map(([value, label]) => ({ value, label })),
  online_delays: [2, 3],
  desired_return_range: [-20, 140],
  default_desired_return: 20,
  temperature_range: [0.8, 1.1],
  default_temperature: 1,
};
