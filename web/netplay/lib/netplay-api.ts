export type Choice = { value: string; label: string };

export type Options = {
  characters: Choice[];
  imitations: Choice[];
  stages: Choice[];
  online_delays: number[];
  max_games: number;
  no_show_seconds: number;
  rematch_seconds: number;
};

export type Capacity = {
  capacity: number;
  active: number;
  queued: number;
};

export type Job = {
  id: string;
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  requested_stage: string | null;
  status: string;
  queue_position: number | null;
  attempt: number;
  game_count: number;
  connect_code: string | null;
  actual_stage: string | null;
  last_result: string | null;
  error_code: string | null;
  connect_deadline: number | null;
  rematch_deadline: number | null;
  cancel_after_game: boolean;
};

export type JobCredentials = Job & { token: string };

export type CreateJob = {
  invite_code: string;
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
};

export type Rematch = {
  character: string;
  imitation: string;
  stage: string;
};

const API_BASE = process.env.NEXT_PUBLIC_HAL_API_URL ?? 'http://localhost:8080';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('Content-Type', 'application/json');
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
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

export function cancelJob(id: string, token: string): Promise<Job> {
  return request(`/v1/jobs/${id}`, authorized(token, { method: 'DELETE' }));
}

export function requestRematch(
  id: string,
  token: string,
  values: Rematch,
): Promise<Job> {
  return request(
    `/v1/jobs/${id}/rematch`,
    authorized(token, { method: 'POST', body: JSON.stringify(values) }),
  );
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
  imitations: [
    ['IBDW#0', 'iBDW'],
    ['ZAIN#0', 'Zain'],
    ['MANG#0', 'Mang0'],
    ['LEFFEN#0', 'Leffen'],
    ['PIPLUP#0', 'Pipsqueak'],
    ['AMSA#0', 'aMSa'],
    ['HBOX#1', 'Hungrybox'],
    ['PLATINUM', 'Platinum rank'],
    ['DIAMOND', 'Diamond rank'],
    ['MASTER', 'Master rank'],
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
  max_games: 5,
  no_show_seconds: 60,
  rematch_seconds: 60,
};
