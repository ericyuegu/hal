'use client';

import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react';

import type { Options } from '@/lib/netplay-api';
import {
  characterItems,
  iconSrc,
  imitationGroups,
  imitationPhrase,
  matchedAlias,
  matches,
  prefixMatch,
} from '@/lib/roster';
import type { Group, Item } from '@/lib/roster';

export type Panel = 'player' | 'char' | 'diff' | 'stage';

/** A blank without a setter renders as fixed text. */
type Slot<T> = { value: T; set?: (value: T) => void };

export function Sentence({
  lead,
  options,
  imitation,
  character,
  difficulty,
  stage,
  open,
  setOpen,
  small,
}: {
  lead: string;
  options: Options;
  imitation: Slot<string>;
  character: Slot<string>;
  difficulty?: Slot<number>;
  stage?: Slot<string>;
  open: Panel | null;
  setOpen: (panel: Panel | null) => void;
  small?: boolean;
}) {
  const blanks = useRef(new Map<Panel, HTMLButtonElement>());
  const setters: Record<Panel, unknown> = {
    player: imitation.set,
    char: character.set,
    diff: difficulty?.set,
    stage: stage?.set,
  };
  const shown = open && setters[open] ? open : null;

  useEffect(() => {
    if (!shown) return;
    function outside(event: MouseEvent) {
      const target = event.target as Element | null;
      if (!target?.closest('.panel, .blank')) setOpen(null);
    }
    document.addEventListener('click', outside);
    return () => document.removeEventListener('click', outside);
  }, [shown, setOpen]);

  function close() {
    const was = shown;
    setOpen(null);
    if (was) blanks.current.get(was)?.focus();
  }

  function blank(panel: Panel, children: ReactNode) {
    if (!setters[panel]) return <span className="blank">{children}</span>;
    return (
      <button
        type="button"
        className="blank"
        ref={(element) => {
          if (element) blanks.current.set(panel, element);
          else blanks.current.delete(panel);
        }}
        aria-expanded={shown === panel}
        onClick={() => (shown === panel ? close() : setOpen(panel))}
      >
        {children}
      </button>
    );
  }

  const phrase = imitationPhrase(imitation.value, options.imitations);
  const characterName =
    options.characters.find((choice) => choice.value === character.value)
      ?.label ?? character.value;
  const icon = iconSrc(character.value);
  const stageName =
    stage &&
    (options.stages.find((choice) => choice.value === stage.value)?.label ??
      stage.value);

  return (
    <>
      <h1 className={small ? 'sentence small' : 'sentence'}>
        {lead}{' '}
        <span className="nw">
          {blank('player', phrase.text)}
          {phrase.possessive && '’s'}
        </span>{' '}
        {blank(
          'char',
          <>
            {icon && (
              // Pixel art at 24px; image optimization would only blur it.
              // oxlint-disable-next-line nextjs/no-img-element
              <img src={icon} alt="" width={24} height={24} />
            )}
            {characterName}
          </>,
        )}
        {difficulty && (
          <>
            {' '}
            at difficulty{' '}
            <span className="nw">
              {blank('diff', <span className="num">{difficulty.value}</span>)}.
            </span>
          </>
        )}
        {stage && (
          <>
            {' '}
            on <span className="nw">{blank('stage', stageName)}.</span>
          </>
        )}
      </h1>
      {shown === 'player' && imitation.set && (
        <PickerPanel
          label="players"
          groups={imitationGroups(options.imitations)}
          selected={imitation.value}
          pick={(value) => {
            imitation.set!(value);
            close();
          }}
          close={close}
        />
      )}
      {shown === 'char' && character.set && (
        <PickerPanel
          label="characters"
          grid
          groups={[{ items: characterItems(options.characters) }]}
          selected={character.value}
          pick={(value) => {
            character.set!(value);
            close();
          }}
          close={close}
        />
      )}
      {shown === 'diff' && difficulty?.set && (
        <DifficultyPanel
          value={difficulty.value}
          set={difficulty.set}
          close={close}
        />
      )}
      {shown === 'stage' && stage?.set && (
        <PickerPanel
          label="stages"
          groups={[
            {
              items: options.stages.map((choice) => ({
                value: choice.value,
                name: choice.label,
                alias: '',
              })),
            },
          ]}
          selected={stage.value}
          pick={(value) => {
            stage.set!(value);
            close();
          }}
          close={close}
        />
      )}
    </>
  );
}

function SearchIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden="true"
    >
      <circle cx="7" cy="7" r="5" />
      <path d="M11 11l4 4" />
    </svg>
  );
}

function PickerPanel({
  label,
  groups,
  grid,
  selected,
  pick,
  close,
}: {
  label: string;
  groups: Group[];
  grid?: boolean;
  selected: string;
  pick: (value: string) => void;
  close: () => void;
}) {
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const all = groups.flatMap((group) => group.items);
  const visible = all.filter((item) => matches(query, item));

  useEffect(() => input.current?.focus(), []);
  useEffect(() => {
    body.current
      ?.querySelector('.opt.hl')
      ?.scrollIntoView({ block: 'nearest' });
  }, [highlight]);

  function search(next: string) {
    setQuery(next);
    const found = all.filter((item) => matches(next, item));
    const best = next
      ? (found.find((item) => prefixMatch(next, item)) ?? found[0])
      : undefined;
    setHighlight(best?.value ?? null);
  }

  function keyDown(event: ReactKeyboardEvent<HTMLElement>) {
    const fromInput = event.target === input.current;
    const columns = grid ? (window.innerWidth <= 720 ? 5 : 9) : 1;
    const step = {
      ArrowDown: columns,
      ArrowUp: -columns,
      ArrowRight: 1,
      ArrowLeft: -1,
    }[event.key];
    const vertical = event.key === 'ArrowDown' || event.key === 'ArrowUp';
    if (step !== undefined && (!fromInput || vertical || (grid && !query))) {
      event.preventDefault();
      const index = visible.findIndex((item) => item.value === highlight);
      const next =
        visible[
          Math.max(
            0,
            Math.min(visible.length - 1, index < 0 ? 0 : index + step),
          )
        ];
      if (!next) return;
      setHighlight(next.value);
      if (!fromInput)
        body.current
          ?.querySelector<HTMLButtonElement>(
            `[data-value="${CSS.escape(next.value)}"]`,
          )
          ?.focus();
    } else if (event.key === 'Enter' && fromInput) {
      event.preventDefault();
      const target =
        visible.find((item) => item.value === highlight) ?? visible[0];
      if (target) pick(target.value);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      close();
    }
  }

  function option(item: Item) {
    const hit = matches(query, item);
    const classes = ['opt'];
    if (item.value === highlight) classes.push('hl');
    if (grid && !hit) classes.push('dim');
    return (
      <button
        key={item.value}
        type="button"
        className={classes.join(' ')}
        data-value={item.value}
        aria-pressed={item.value === selected}
        onClick={() => pick(item.value)}
        onFocus={() => setHighlight(item.value)}
        onKeyDown={keyDown}
      >
        {item.icon && (
          // Pixel art at 36px; image optimization would only blur it.
          // oxlint-disable-next-line nextjs/no-img-element
          <img src={item.icon} alt="" width={36} height={36} />
        )}
        <span>{item.name}</span>
        {!grid && <span className="alias">{matchedAlias(query, item)}</span>}
      </button>
    );
  }

  return (
    <div className="panel">
      <label className="search">
        <SearchIcon />
        <input
          ref={input}
          value={query}
          onChange={(event) => search(event.target.value)}
          placeholder={`Search ${label}`}
          aria-label={`Search ${label}`}
          autoComplete="off"
          spellCheck={false}
          onKeyDown={keyDown}
        />
        <span className="keys">
          <kbd>{grid ? '←→' : '↑↓'}</kbd> move <kbd>↵</kbd> pick <kbd>esc</kbd>{' '}
          close
        </span>
      </label>
      <div className="body" ref={body}>
        {grid ? (
          <div className="cgrid">{all.map(option)}</div>
        ) : (
          groups.map((group, index) => {
            const items = group.items.filter((item) => matches(query, item));
            if (items.length === 0) return null;
            return (
              <div key={group.title ?? index}>
                {group.title && <div className="group">{group.title}</div>}
                <div className="plist">{items.map(option)}</div>
              </div>
            );
          })
        )}
        {visible.length === 0 && <div className="empty">No match</div>}
      </div>
    </div>
  );
}

function DifficultyPanel({
  value,
  set,
  close,
}: {
  value: number;
  set: (value: number) => void;
  close: () => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const range = useRef<HTMLInputElement>(null);
  useEffect(() => range.current?.focus(), []);

  function keyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
    if (event.key !== 'Escape' && event.key !== 'Enter') return;
    event.preventDefault();
    event.stopPropagation();
    close();
  }

  return (
    <div className="panel">
      <div className="diff">
        <div className="diffhead">
          <span className="lbl">Difficulty</span>
          <input
            className="lv"
            inputMode="numeric"
            aria-label="Difficulty value"
            value={draft ?? String(value)}
            onChange={(event) => {
              setDraft(event.target.value);
              const next = Number.parseInt(event.target.value, 10);
              if (Number.isFinite(next)) set(Math.max(0, Math.min(100, next)));
            }}
            onBlur={() => setDraft(null)}
            onKeyDown={keyDown}
          />
        </div>
        <div className="bar">
          <div className="f" style={{ width: `${value}%` }} />
          <div className="k" style={{ left: `${value}%` }} />
          <input
            ref={range}
            type="range"
            min={0}
            max={100}
            step={1}
            value={value}
            aria-label="Difficulty"
            onChange={(event) => set(Number(event.target.value))}
            onKeyDown={keyDown}
          />
        </div>
        <div className="ends">
          <span>0 · chill</span>
          <span>locked-in · 100</span>
        </div>
      </div>
    </div>
  );
}

type Hotkeys = Partial<Record<string, () => void>>;

function typing(element: Element | null): boolean {
  if (element instanceof HTMLTextAreaElement) return true;
  return (
    element instanceof HTMLInputElement &&
    !['range', 'checkbox', 'radio'].includes(element.type)
  );
}

/**
 * Page hotkeys, keyed by lowercase key plus 'enter', 'mod+enter', and
 * 'escape'. Text fields keep their keys, except Esc and Ctrl/Cmd+Enter.
 */
export function useHotkeys(keys: Hotkeys) {
  const current = useRef(keys);
  useEffect(() => {
    current.current = keys;
  });
  useEffect(() => {
    function keyDown(event: KeyboardEvent) {
      const actions = current.current;
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        const action = actions['mod+enter'];
        if (!action) return;
        event.preventDefault();
        action();
        return;
      }
      if (event.key === 'Escape') {
        actions.escape?.();
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (typing(document.activeElement)) return;
      if (event.key === 'Enter') {
        const action = actions.enter;
        if (!action || document.activeElement !== document.body) return;
        event.preventDefault();
        action();
        return;
      }
      const action = actions[event.key.toLowerCase()];
      if (!action) return;
      event.preventDefault();
      action();
    }
    document.addEventListener('keydown', keyDown);
    return () => document.removeEventListener('keydown', keyDown);
  }, []);
}

export function ShortcutSheet({
  shortcuts,
  close,
}: {
  shortcuts: [ReactNode, string][];
  close: () => void;
}) {
  return (
    <div className="sheet">
      <button
        type="button"
        className="sheet-backdrop"
        aria-label="Close shortcuts"
        onClick={close}
      />
      <dialog open className="sheet-card" aria-label="Keyboard shortcuts">
        <div className="lbl sheet-title">Shortcuts</div>
        <dl>
          {shortcuts.map(([keys, action]) => (
            <div key={action} className="sheet-row">
              <dt>{keys}</dt>
              <dd>{action}</dd>
            </div>
          ))}
        </dl>
      </dialog>
    </div>
  );
}
