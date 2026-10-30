/* HAL policy boundary over the simulator's public ABI. No gameplay lives here. */
#include "api.h"
#include "hal_native_schema.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HAL_ABI 2
#define HAL_CORE_ABI 2

typedef struct {
  uint32_t columns[HAL_COLUMN_COUNT];
  int32_t frame_id;
  float applied_action[2][14];
  int16_t wire_inputs[2][7];
  float reward[2];
  uint8_t terminated;
  uint8_t truncated;
  uint8_t reset;
  uint8_t padding;
} HalFrame;
_Static_assert(sizeof(HalFrame) == 440, "HAL frame layout");
_Static_assert(sizeof(MslMatchConfig) == 52, "native config ABI 2");
_Static_assert(sizeof(MslObservation) == 1240, "native observation ABI 2");
_Static_assert(sizeof(MslTerminal) == 16, "native terminal ABI 2");

typedef struct {
  void *library;
  MslBatch *batch;
  uint32_t count;
  HalFrame *frames;
  MslInput *inputs;
  float (*actions)[2][14];
  int16_t (*wire)[2][7];
  MslObservation *observations;
  MslTerminal *terminals;
  MslMatchConfig *configs;
  uint8_t *active;
  uint8_t *mask;
  MslResult (*create)(const char *, uint32_t, MslBatch **);
  void (*destroy)(MslBatch *);
  MslResult (*reset)(MslBatch *, const MslMatchConfig *, const uint8_t *, MslObservation *);
  MslResult (*step)(MslBatch *, const MslInput *, const uint8_t *, MslObservation *, MslTerminal *);
  MslResult (*observe)(const MslBatch *, MslObservation *, MslTerminal *);
  MslResult (*save_size)(const MslBatch *, uint32_t, size_t *);
  MslResult (*save)(const MslBatch *, uint32_t, void *, size_t, size_t *);
  MslResult (*restore)(MslBatch *, uint32_t, const void *, size_t);
} HalBatch;

uint32_t hal_native_abi(void) {
  return HAL_ABI;
}

const char *hal_native_schema(void) {
  return HAL_SCHEMA_HASH;
}

size_t hal_native_frame_size(void) {
  return sizeof(HalFrame);
}

const char *hal_native_error(int code) {
  switch (code) {
    case 0:
      return "success";
    case 1:
      return "invalid native argument";
    case 2:
      return "out of memory";
    case 3:
      return "invalid native state";
    case 4:
      return "incompatible native state";
    case 10:
      return "cannot step a lane before reset";
    case 11:
      return "reset terminal lanes before stepping them";
    case 12:
      return "actions must be finite";
    case 13:
      return "actions are out of range";
    case 14:
      return "invalid adapter buffer or configuration";
    case 15:
      return "native stage does not match HAL match";
    case 16:
      return "no unique roster slot for controller port";
    case 17:
      return "native simulator ABI 2 is required";
    case 18:
      return "cannot load native library or required public symbols";
    default:
      return "unknown native adapter error";
  }
}

void hal_native_destroy(HalBatch *b) {
  if (!b)
    return;
  if (b->batch)
    b->destroy(b->batch);
  if (b->library)
    dlclose(b->library);
  free(b->actions);
  free(b->wire);
  free(b->inputs);
  free(b->observations);
  free(b->terminals);
  free(b->configs);
  free(b->active);
  free(b->mask);
  free(b);
}

int hal_native_create(const char *library, const char *data, uint32_t count,
                      HalFrame *frames, HalBatch **out) {
  if (!library || !data || !count || !frames || !out)
    return 14;
  *out = NULL;
  HalBatch *b = calloc(1, sizeof(*b));
  if (!b)
    return 2;
  b->library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!b->library) {
    hal_native_destroy(b);
    return 18;
  }
  uint32_t (*abi)(void) = dlsym(b->library, "msl_api_abi_version");
  if (!abi || abi() != HAL_CORE_ABI) {
    hal_native_destroy(b);
    return 17;
  }
#define LOAD(field, name)                  \
  do {                                     \
    b->field = dlsym(b->library, name);    \
    if (!b->field) {                       \
      hal_native_destroy(b);              \
      return 18;                          \
    }                                      \
  } while (0)
  LOAD(create, "msl_batch_create");
  LOAD(destroy, "msl_batch_destroy");
  LOAD(reset, "msl_batch_reset");
  LOAD(step, "msl_batch_step_masked");
  LOAD(observe, "msl_batch_observe");
  LOAD(save_size, "msl_batch_save_size");
  LOAD(save, "msl_batch_save");
  LOAD(restore, "msl_batch_restore");
#undef LOAD
  b->count = count;
  b->frames = frames;
  b->inputs = calloc(count, sizeof(*b->inputs));
  b->actions = calloc(count, sizeof(*b->actions));
  b->wire = calloc(count, sizeof(*b->wire));
  b->observations = calloc(count, sizeof(*b->observations));
  b->terminals = calloc(count, sizeof(*b->terminals));
  b->configs = calloc(count, sizeof(*b->configs));
  b->active = calloc(count, 1);
  b->mask = calloc(count, 1);
  if (!b->actions || !b->wire || !b->inputs || !b->observations ||
      !b->terminals || !b->configs || !b->active || !b->mask) {
    hal_native_destroy(b);
    return 2;
  }
  int rc = b->create(data, count, &b->batch);
  if (rc) {
    hal_native_destroy(b);
    return rc;
  }
  /* The public step/observe API reads every slot, including masked slots.
     Initialize private slots once; HAL still requires an explicit reset before
     a lane can step and keeps its public columns masked until then. */
  MslMatchConfig (*default_config)(void) = dlsym(b->library, "msl_match_config_default");
  if (!default_config) {
    hal_native_destroy(b);
    return 18;
  }
  for (uint32_t i = 0; i < count; ++i)
    b->configs[i] = default_config();
  rc = b->reset(b->batch, b->configs, NULL, b->observations);
  if (rc) {
    hal_native_destroy(b);
    return rc;
  }
  memset(frames, 0, count * sizeof(*frames));
  for (uint32_t i = 0; i < count; ++i)
    memcpy(frames[i].columns, hal_masks, sizeof(hal_masks));
  *out = b;
  return 0;
}
static int hal_stage(uint32_t stage) {
  switch (stage) {
    case 2:
      return 8;
    case 3:
      return 18;
    case 8:
      return 6;
    case 28:
      return 26;
    case 31:
      return 24;
    case 32:
      return 25;
    default:
      return -1;
  }
}

static void put_float(uint32_t *word, float value) {
  memcpy(word, &value, 4);
}

static void player_columns(HalFrame *f, int base, const MslObservationPlayer *p, int leader) {
  if (!p->present) {
    memcpy(f->columns + base, hal_masks + base, 11 * 4);
    if (leader)
      f->columns[base + 7] = p->stocks;
    return;
  }
  put_float(f->columns + base, p->pos_x);
  put_float(f->columns + base + 1, p->pos_y);
  put_float(f->columns + base + 2, p->percent);
  put_float(f->columns + base + 3, p->shield_hp);
  put_float(f->columns + base + 4, p->facing ? 1.0f : -1.0f);
  put_float(f->columns + base + 5, p->hitlag_left);
  f->columns[base + 6] = p->action_id;
  f->columns[base + 7] = p->stocks;
  f->columns[base + 8] = p->jumps_left;
  f->columns[base + 9] = p->hurtbox_state;
  f->columns[base + 10] = !p->on_ground;
}

static int project(HalBatch *b, uint32_t lane) {
  const MslObservation *obs = b->observations + lane;
  const MslMatchConfig *cfg = b->configs + lane;
  HalFrame *f = b->frames + lane;
  if (obs->stage_id != cfg->stage || hal_stage(obs->stage_id) < 0)
    return 15;
  int slots[2] = {-1, -1};
  for (int roster = 0; roster < 2; ++roster) {
    int count = 0;
    for (int slot = 0; slot < MSL_MAX_PLAYERS; ++slot) {
      if (obs->slots[slot].source_player == roster) {
        slots[roster] = slot;
        ++count;
      }
    }
    if (count != 1)
      return 16;
  }
  f->frame_id = obs->frame_id;
  f->columns[HAL_COL_STAGE] = hal_stage(obs->stage_id);
  for (int roster = 0; roster < 2; ++roster) {
    int port = cfg->players[roster].controller_port;
    int base = port == 0 ? HAL_COL_P1_POSITION_X : HAL_COL_P2_POSITION_X;
    f->columns[HAL_COL_P1_CHARACTER + port] = cfg->players[roster].character;
    player_columns(f, base, obs->slots + slots[roster], 1);
    player_columns(f, base + 11, obs->followers + slots[roster], 0);
  }
  /* Stable insertion of the four smallest IDs preserves array order on ties. */
  int ordered[4] = {-1, -1, -1, -1};
  for (int i = 0; i < MSL_MAX_ITEMS; ++i) {
    if (!obs->items[i].exists)
      continue;
    for (int j = 0; j < 4; ++j) {
      if (ordered[j] < 0 || obs->items[i].spawn_id < obs->items[ordered[j]].spawn_id) {
        for (int k = 3; k > j; --k)
          ordered[k] = ordered[k - 1];
        ordered[j] = i;
        break;
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    int base = HAL_COL_ITEM0_TYPE + 6 * i;
    if (ordered[i] < 0) {
      memcpy(f->columns + base, hal_masks + base, 6 * 4);
      continue;
    }
    const MslItem *item = obs->items + ordered[i];
    f->columns[base] = item->type;
    f->columns[base + 1] = item->state;
    put_float(f->columns + base + 2, item->pos_x);
    put_float(f->columns + base + 3, item->pos_y);
    put_float(f->columns + base + 4, item->vel_x);
    put_float(f->columns + base + 5, item->vel_y);
  }
  return 0;
}
/* Explicit ties-to-even, independent of the caller's rounding mode for integer conversion. */
static int rounded(double x) {
  double lower = floor(x);
  double delta = x - lower;
  int n = (int)lower;
  return n + (delta > 0.5 || (delta == 0.5 && n % 2 != 0));
}

static void pack(const float actions[14], int16_t wire[7], MslInputPlayer *out) {
  for (int i = 0; i < 4; ++i) {
    double x = actions[i];
    wire[i] = rounded(((x + 1.0) / 2.0 - 0.5) * 160.0);
  }
  for (int i = 4; i < 6; ++i) {
    int v = rounded((double)actions[i] * 140.0);
    wire[i] = v;
  }
  unsigned buttons = 0;
  for (int i = 0; i < 8; ++i) {
    if (actions[6 + i] > 0.5f)
      buttons |= hal_buttons[i];
  }
  /* Dolphin's GCPad digital shoulder sets the corresponding analog axis.
     Slippi publishes that axis after the game's 140-byte trigger clamp. */
  if (buttons & MSL_BUTTON_L) wire[4] = 140;
  if (buttons & MSL_BUTTON_R) wire[5] = 140;
  wire[6] = buttons;
  out->main_x = wire[0];
  out->main_y = wire[1];
  out->c_x = wire[2];
  out->c_y = wire[3];
  out->l = wire[4];
  out->r = wire[5];
  out->buttons = buttons;
}

int hal_native_quantize(const float *actions, int16_t *wire, size_t players) {
  if (!actions || !wire)
    return 14;
  for (size_t p = 0; p < players; ++p) {
    for (int i = 0; i < 14; ++i) {
      float v = actions[p * 14 + i];
      if (!isfinite(v))
        return 12;
      if (v < (i < 4 ? -1.0f : 0.0f) || v > 1.0f)
        return 13;
    }
  }
  MslInputPlayer ignored;
  for (size_t p = 0; p < players; ++p)
    pack(actions + p * 14, wire + p * 7, &ignored);
  return 0;
}

int hal_native_reset(HalBatch *b, const MslMatchConfig *configs, const uint8_t *mask) {
  if (!b || !configs || !mask)
    return 14;
  for (uint32_t i = 0; i < b->count; ++i) {
    if (mask[i]) {
      const MslMatchConfig *c = configs + i;
      if (c->num_players != 2 || hal_stage(c->stage) < 0 ||
          c->players[0].controller_port < 0 || c->players[0].controller_port > 1 ||
          c->players[1].controller_port != 1 - c->players[0].controller_port)
        return 14;
    }
  }
  int rc = b->reset(b->batch, configs, mask, b->observations);
  if (rc)
    return rc;
  for (uint32_t i = 0; i < b->count; ++i) {
    HalFrame *f = b->frames + i;
    f->reward[0] = f->reward[1] = 0;
    f->reset = !!mask[i];
    if (mask[i]) {
      b->configs[i] = configs[i];
      b->active[i] = 1;
      memset(f->applied_action, 0, sizeof(f->applied_action));
      memset(f->wire_inputs, 0, sizeof(f->wire_inputs));
      f->terminated = f->truncated = 0;
    }
    if (b->active[i] && (rc = project(b, i)))
      return rc;
  }
  return 0;
}

static int finish_step(HalBatch *b) {
  int rc;
  for (uint32_t i = 0; i < b->count; ++i) {
    HalFrame *f = b->frames + i;
    int old_stock[2] = {f->columns[HAL_COL_P1_STOCK], f->columns[HAL_COL_P2_STOCK]};
    f->reward[0] = f->reward[1] = 0;
    f->reset = 0;
    if (b->active[i] && (rc = project(b, i)))
      return rc;
    if (!b->mask[i])
      continue;
    memcpy(f->applied_action, b->actions[i], sizeof(f->applied_action));
    memcpy(f->wire_inputs, b->wire[i], sizeof(f->wire_inputs));
    int stock[2] = {f->columns[HAL_COL_P1_STOCK], f->columns[HAL_COL_P2_STOCK]};
    int loss[2];
    int final[2];
    for (int p = 0; p < 2; ++p) {
      loss[p] = stock[p] < old_stock[p] && stock[p] != INT32_MAX && old_stock[p] != INT32_MAX;
      final[p] = loss[p] && stock[p] == 0;
    }
    /* Stocks are the whole reward; damage counts only through the stocks it wins. */
    float base = 120 * (loss[1] - loss[0]) + 50 * (final[1] - final[0]);
    f->reward[0] = base;
    /* Subtracting from zero keeps a zero reward at +0, matching the canonical bytes. */
    f->reward[1] = 0.0f - base;
    f->terminated = b->terminals[i].stockout != 0;
    f->truncated = b->terminals[i].max_frame_reached != 0 && !f->terminated;
  }
  return 0;
}

int hal_native_step(HalBatch *b, const char *actions, ptrdiff_t lane_stride,
                    ptrdiff_t port_stride, ptrdiff_t channel_stride,
                    const uint8_t *mask, ptrdiff_t mask_stride) {
  if (!b || !actions || !mask)
    return 14;
  /* Validate the whole selected batch before changing any output or simulator state. */
  for (uint32_t i = 0; i < b->count; ++i) {
    if (*(mask + (ptrdiff_t)i * mask_stride)) {
      if (!b->active[i])
        return 10;
      if (b->frames[i].terminated || b->frames[i].truncated)
        return 11;
      for (int p = 0; p < 2; ++p) {
        for (int c = 0; c < 14; ++c) {
          float v;
          memcpy(&v, actions + (ptrdiff_t)i * lane_stride + p * port_stride + c * channel_stride, 4);
          if (!isfinite(v))
            return 12;
          if (v < (c < 4 ? -1.0f : 0.0f) || v > 1.0f)
            return 13;
        }
      }
    }
  }
  for (uint32_t i = 0; i < b->count; ++i) {
    b->mask[i] = !!*(mask + (ptrdiff_t)i * mask_stride);
    if (!b->mask[i])
      continue;
    memset(b->inputs + i, 0, sizeof(*b->inputs));
    for (int roster = 0; roster < 2; ++roster) {
      int port = b->configs[i].players[roster].controller_port;
      float *a = b->actions[i][port];
      for (int c = 0; c < 14; ++c)
        memcpy(a + c, actions + (ptrdiff_t)i * lane_stride + port * port_stride + c * channel_stride, 4);
      pack(a, b->wire[i][port], &b->inputs[i].players[roster]);
    }
  }
  int rc = b->step(b->batch, b->inputs, b->mask, b->observations, b->terminals);
  if (rc)
    return rc;
  return finish_step(b);
}
int hal_native_replay_sizes(size_t sizes[3]) {
  if (!sizes) return 14;
  sizes[0] = sizeof(MslReplayConfig);
  sizes[1] = sizeof(MslReplayInput);
  sizes[2] = sizeof(MslReplayFrame);
  return 0;
}

int hal_native_reset_replay(HalBatch *b, const MslReplayConfig *configs,
                            const MslReplayInput *previous, const MslReplayFrame *first,
                            const MslMatchConfig *matches) {
  if (!b || !configs || !previous || !first || !matches) return 14;
  uint32_t (*abi)(void) = dlsym(b->library, "msl_replay_abi_version");
  MslResult (*reset)(MslBatch *, const MslReplayConfig *, const MslReplayInput *, const uint8_t *) =
      dlsym(b->library, "msl_batch_reset_replay");
  MslResult (*step)(MslBatch *, const MslReplayFrame *, const uint8_t *, MslObservation *, MslTerminal *) =
      dlsym(b->library, "msl_batch_step_replay");
  if (!abi || abi() != 1 || !reset || !step) return 18;
  for (uint32_t i = 0; i < b->count; ++i) {
    if (matches[i].num_players != 2 || configs[i].num_players != 2 ||
        matches[i].stage != configs[i].stage_id || matches[i].is_teams || configs[i].is_teams ||
        matches[i].random_seed != configs[i].initial_random_seed ||
        matches[i].stocks != configs[i].stock_count ||
        matches[i].damage_ratio != configs[i].match_damage_ratio ||
        matches[i].ucf_cardinals != configs[i].ucf_cardinals_1_0_enabled ||
        matches[i].max_frame != -1 ||
        matches[i].players[0].controller_port < 0 || matches[i].players[0].controller_port > 1 ||
        matches[i].players[1].controller_port != 1 - matches[i].players[0].controller_port)
      return 14;
    for (int p = 0; p < 2; ++p) {
      if (matches[i].players[p].character != configs[i].players[p].char_id ||
          matches[i].players[p].facing != MSL_FACING_AUTO ||
          matches[i].players[p].costume != configs[i].players[p].costume_id ||
          matches[i].players[p].handicap != configs[i].players[p].handicap ||
          matches[i].players[p].start_percent != configs[i].players[p].start_percent)
        return 14;
    }
  }
  int rc = reset(b->batch, configs, previous, NULL);
  if (rc) return rc;
  rc = step(b->batch, first, NULL, b->observations, b->terminals);
  if (rc) return rc;
  for (uint32_t i = 0; i < b->count; ++i) {
    b->configs[i] = matches[i];
    b->active[i] = 1;
    memset(b->frames + i, 0, sizeof(*b->frames));
    b->frames[i].reset = 1;
    if ((rc = project(b, i))) return rc;
    for (int roster = 0; roster < 2; ++roster) {
      const MslReplayInputPlayer *input = &first[i].input.p[roster];
      int16_t *wire = b->frames[i].wire_inputs[matches[i].players[roster].controller_port];
      wire[0] = input->main_x; wire[1] = input->main_y;
      wire[2] = input->c_x; wire[3] = input->c_y;
      wire[4] = input->l; wire[5] = input->r; wire[6] = input->buttons;
    }
    b->frames[i].terminated = b->terminals[i].stockout != 0;
  }
  return 0;
}

int hal_native_step_replay(HalBatch *b, const MslReplayFrame *frames, const uint8_t *mask) {
  if (!b || !frames || !mask) return 14;
  MslResult (*step)(MslBatch *, const MslReplayFrame *, const uint8_t *, MslObservation *, MslTerminal *) =
      dlsym(b->library, "msl_batch_step_replay");
  if (!step) return 18;
  for (uint32_t i = 0; i < b->count; ++i) {
    b->mask[i] = !!mask[i];
    if (!mask[i]) continue;
    if (!b->active[i]) return 10;
    if (b->frames[i].terminated || b->frames[i].truncated) return 11;
    memset(b->actions[i], 0, sizeof(b->actions[i]));
    for (int roster = 0; roster < 2; ++roster) {
      const MslReplayInputPlayer *input = &frames[i].input.p[roster];
      int16_t *wire = b->wire[i][b->configs[i].players[roster].controller_port];
      wire[0] = input->main_x; wire[1] = input->main_y;
      wire[2] = input->c_x; wire[3] = input->c_y;
      wire[4] = input->l; wire[5] = input->r; wire[6] = input->buttons;
    }
  }
  int rc = step(b->batch, frames, mask, b->observations, b->terminals);
  return rc ? rc : finish_step(b);
}

int hal_native_save_size(HalBatch *b, uint32_t lane, size_t *size) {
  if (!b || lane >= b->count || !b->active[lane])
    return 14;
  return b->save_size(b->batch, lane, size);
}

int hal_native_save(HalBatch *b, uint32_t lane, void *data, size_t size, size_t *written) {
  if (!b || lane >= b->count || !b->active[lane])
    return 14;
  return b->save(b->batch, lane, data, size, written);
}

int hal_native_restore_many(HalBatch *b, const uint32_t *lanes,
                            const void *const *data, const size_t *sizes,
                            const MslMatchConfig *configs, const HalFrame *saved,
                            uint32_t count) {
  if (!b || !lanes || !data || !sizes || !configs || !saved || count > b->count)
    return 14;
  for (uint32_t j = 0; j < count; ++j) {
    if (lanes[j] >= b->count || !data[j] || !sizes[j]) return 14;
    if (configs[j].num_players != 2 || configs[j].is_teams || hal_stage(configs[j].stage) < 0 ||
        configs[j].players[0].controller_port < 0 || configs[j].players[0].controller_port > 1 ||
        configs[j].players[1].controller_port != 1 - configs[j].players[0].controller_port)
      return 14;
    for (uint32_t k = 0; k < j; ++k)
      if (lanes[j] == lanes[k]) return 14;
  }
  int rc;
  for (uint32_t j = 0; j < count; ++j) {
    rc = b->restore(b->batch, lanes[j], data[j], sizes[j]);
    if (rc) return rc;
    b->configs[lanes[j]] = configs[j];
    b->active[lanes[j]] = 1;
    b->frames[lanes[j]] = saved[j];
  }
  rc = b->observe(b->batch, b->observations, b->terminals);
  if (rc) return rc;
  for (uint32_t i = 0; i < b->count; ++i) {
    b->frames[i].reward[0] = b->frames[i].reward[1] = 0;
    b->frames[i].reset = 0;
    if (b->active[i] && (rc = project(b, i))) return rc;
  }
  return 0;
}
