import type { Choice } from '@/lib/netplay-api';

// Page-side display data. The policy's /v1/options owns which values exist;
// these tables only add icons, nicknames, and sentence grammar.

// Stock icons in public/characters, fetched from ssb.wiki.gallery (*HeadSSBM.png).
const ICONS = new Set([
  'DOC',
  'MARIO',
  'LUIGI',
  'BOWSER',
  'PEACH',
  'YOSHI',
  'DK',
  'CPTFALCON',
  'GANONDORF',
  'FALCO',
  'FOX',
  'NESS',
  'POPO',
  'KIRBY',
  'SAMUS',
  'ZELDA',
  'LINK',
  'YOUNG_LINK',
  'PICHU',
  'PIKACHU',
  'JIGGLYPUFF',
  'MEWTWO',
  'GAMEANDWATCH',
  'MARTH',
  'ROY',
  'SHEIK',
]);

// Character-select order, so the grid reads like the game's.
const CHARACTER_ORDER = [
  'DOC',
  'MARIO',
  'LUIGI',
  'BOWSER',
  'PEACH',
  'YOSHI',
  'DK',
  'CPTFALCON',
  'GANONDORF',
  'FALCO',
  'FOX',
  'NESS',
  'POPO',
  'KIRBY',
  'SAMUS',
  'ZELDA',
  'LINK',
  'YOUNG_LINK',
  'PICHU',
  'PIKACHU',
  'JIGGLYPUFF',
  'MEWTWO',
  'GAMEANDWATCH',
  'MARTH',
  'ROY',
  'SHEIK',
];

const CHARACTER_ALIASES: Record<string, string> = {
  DOC: 'doc',
  LUIGI: 'weegee',
  BOWSER: 'koopa',
  DK: 'dk',
  CPTFALCON: 'cf falcon',
  GANONDORF: 'ganon',
  FALCO: 'bird',
  FOX: 'spacie',
  POPO: 'ic ics icies popo nana',
  YOUNG_LINK: 'yl ylink',
  PIKACHU: 'pika',
  JIGGLYPUFF: 'puff jiggs',
  MEWTWO: 'm2',
  GAMEANDWATCH: 'gnw g&w gw',
  SHEIK: 'shiek',
};

// Keyed by the player's label, which is stable across policy vocabularies.
const PLAYER_ALIASES: Record<string, string> = {
  BillyBoPeep: 'billy',
  BobbyBigBallz: 'bobby',
  Cody: 'cody schwab',
  DesertSnoopy: 'snoopy',
  DruggedFox: 'drugged',
  FknSilver: 'silver',
  Grab2Win: 'g2w',
  Hungrybox: 'hbox',
  iBDW: 'cody schwab',
  ILikeTurtles: 'turtles',
  JahRidin: 'jah',
  M2K: 'mew2king',
  Mang0: 'mango',
  Monotheon: 'mono',
  'mr. dokie': 'moky',
  Pipsqueak: 'plup piplup',
  Siddward: 'sidd',
  Solobattle: 'solo',
  TechnoSpider: 'techno',
};

const RANKS: Record<string, string> = {
  MASTER: 'Master',
  DIAMOND: 'Diamond',
  PLATINUM: 'Platinum',
};
const RANK_ORDER = ['MASTER', 'DIAMOND', 'PLATINUM'];
const ANYONE = 'MASKED';

export type Item = {
  value: string;
  name: string;
  alias: string;
  icon?: string;
};
export type Group = { title?: string; items: Item[] };

export function iconSrc(character: string): string | undefined {
  return ICONS.has(character) ? `/characters/${character}.png` : undefined;
}

export function characterItems(choices: Choice[]): Item[] {
  const position = (value: string) => {
    const index = CHARACTER_ORDER.indexOf(value);
    return index < 0 ? CHARACTER_ORDER.length : index;
  };
  return [...choices]
    .sort((a, b) => position(a.value) - position(b.value))
    .map((choice) => ({
      value: choice.value,
      name: choice.label,
      alias: CHARACTER_ALIASES[choice.value] ?? '',
      icon: iconSrc(choice.value),
    }));
}

export function imitationGroups(choices: Choice[]): Group[] {
  const offered = new Set(choices.map((choice) => choice.value));
  const ranks: Item[] = RANK_ORDER.filter((value) => offered.has(value)).map(
    (value) => ({ value, name: RANKS[value], alias: '' }),
  );
  if (offered.has(ANYONE))
    ranks.push({ value: ANYONE, name: 'Anyone', alias: 'none' });
  const players = choices
    .filter((choice) => !(choice.value in RANKS) && choice.value !== ANYONE)
    .map((choice) => ({
      value: choice.value,
      name: choice.label,
      alias: PLAYER_ALIASES[choice.label] ?? '',
    }))
    .sort((a, b) =>
      a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }),
    );
  return [
    { title: 'Slippi ranks', items: ranks },
    { title: 'Players', items: players },
  ].filter((group) => group.items.length > 0);
}

export function defaultImitation(choices: Choice[]): string {
  return choices.some((choice) => choice.value === 'MASTER')
    ? 'MASTER'
    : choices[0].value;
}

/** Ranks read as an adjective ("Master-rank Falco"); players and "anyone" take a possessive. */
export function imitationPhrase(
  value: string,
  choices: Choice[],
): { text: string; possessive: boolean } {
  if (value in RANKS)
    return { text: `${RANKS[value]}-rank`, possessive: false };
  if (value === ANYONE) return { text: 'anyone', possessive: true };
  const label = choices.find((choice) => choice.value === value)?.label;
  return { text: label ?? value, possessive: true };
}

export function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9&]/g, '');
}

function words(item: Item): string[] {
  return [item.name, ...item.alias.split(' ')].filter(Boolean).map(normalize);
}

export function matches(query: string, item: Item): boolean {
  const q = normalize(query);
  return !q || words(item).some((word) => word.includes(q));
}

/** The nickname that matched, when the name itself did not. */
export function matchedAlias(query: string, item: Item): string {
  const q = normalize(query);
  if (!q || normalize(item.name).includes(q)) return '';
  return (
    item.alias.split(' ').find((word) => normalize(word).includes(q)) ?? ''
  );
}

export function prefixMatch(query: string, item: Item): boolean {
  const q = normalize(query);
  return words(item).some((word) => word.startsWith(q));
}

// Difficulty is a 0-100 linear rescale of the policy's desired_return range.
// The API and stored preferences keep raw desired_return, so a rescale never
// shifts a saved setting.
export function toDifficulty(desired: number, [low, high]: [number, number]) {
  const value = Math.round(((desired - low) / (high - low)) * 100);
  return Math.max(0, Math.min(100, value));
}

export function toReturn(difficulty: number, [low, high]: [number, number]) {
  return Math.round((low + (difficulty / 100) * (high - low)) * 100) / 100;
}

export function clampReturn(desired: number, [low, high]: [number, number]) {
  return Math.max(low, Math.min(high, desired));
}
