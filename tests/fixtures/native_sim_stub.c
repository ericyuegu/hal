#include "msl_api.h"

#include <stdlib.h>
#include <string.h>

typedef struct TestLane {
  MslMatchConfig config;
  MslObservation observation;
  MslTerminal terminal;
  int32_t steps;
  uint8_t active;
} TestLane;

struct MslBatch {
  uint32_t size;
  TestLane lanes[];
};

typedef struct RewardOverride {
  uint8_t enabled;
  uint8_t old_stock[2];
  uint8_t new_stock[2];
  float old_percent[2];
  float new_percent[2];
} RewardOverride;

static uint32_t live_batches;
static RewardOverride reward_override[64];
static uint8_t disappear_on_step[64];
static uint8_t bad_roster_on_reset[64];
static MslInput last_input[64];

void msl_test_set_reward(uint32_t lane, uint8_t old0, uint8_t old1,
                         uint8_t new0, uint8_t new1, float oldp0,
                         float oldp1, float newp0, float newp1) {
  if (lane >= 64) return;
  RewardOverride* control = &reward_override[lane];
  control->enabled = 1;
  control->old_stock[0] = old0;
  control->old_stock[1] = old1;
  control->new_stock[0] = new0;
  control->new_stock[1] = new1;
  control->old_percent[0] = oldp0;
  control->old_percent[1] = oldp1;
  control->new_percent[0] = newp0;
  control->new_percent[1] = newp1;
}

void msl_test_disappear_on_step(uint32_t lane) {
  if (lane < 64) disappear_on_step[lane] = 1;
}

void msl_test_bad_roster_on_reset(uint32_t lane) {
  if (lane < 64) bad_roster_on_reset[lane] = 1;
}

void msl_test_copy_last_input(uint32_t lane, void* out) {
  if (lane < 64 && out) memcpy(out, &last_input[lane], sizeof(MslInput));
}

const char* msl_result_string(MslResult result) {
  switch (result) {
    case MSL_OK: return "ok";
    case MSL_INVALID_ARGUMENT: return "invalid argument";
    case MSL_OUT_OF_MEMORY: return "out of memory";
    case MSL_INVALID_STATE: return "invalid state";
    case MSL_INCOMPATIBLE: return "incompatible";
    default: return "unknown";
  }
}

uint32_t msl_api_abi_version(void) { return 2; }

MslMatchConfig msl_match_config_default(void) {
  MslMatchConfig config = {0};
  config.stage = MSL_STAGE_FINAL_DESTINATION;
  config.max_frame = -1;
  config.damage_ratio = 1.0f;
  config.stocks = 4;
  config.num_players = 2;
  return config;
}

MslResult msl_game_data_acquire(const char* data_root) {
  return data_root ? MSL_OK : MSL_INVALID_ARGUMENT;
}
void msl_game_data_release(void) {}
uint32_t msl_game_data_references(void) { return live_batches; }

MslResult msl_batch_create(const char* data_root, uint32_t batch_size,
                           MslBatch** out_batch) {
  if (!data_root || !batch_size || batch_size > 64 || !out_batch) {
    return MSL_INVALID_ARGUMENT;
  }
  MslBatch* batch = calloc(1, sizeof(*batch) + batch_size * sizeof(TestLane));
  if (!batch) return MSL_OUT_OF_MEMORY;
  batch->size = batch_size;
  ++live_batches;
  *out_batch = batch;
  return MSL_OK;
}

void msl_batch_destroy(MslBatch* batch) {
  if (batch) {
    --live_batches;
    free(batch);
  }
}
uint32_t msl_batch_size(const MslBatch* batch) { return batch ? batch->size : 0; }

static void reset_lane(TestLane* lane, const MslMatchConfig* config,
                       uint32_t index) {
  memset(lane, 0, sizeof(*lane));
  lane->config = *config;
  lane->active = 1;
  lane->observation.frame_id = -123;
  lane->observation.stage_id = config->stage;
  lane->observation.num_players = 2;
  lane->observation.viewpoint_player = config->viewpoint_player;
  for (int slot = 0; slot < MSL_MAX_PLAYERS; ++slot) {
    lane->observation.slots[slot].source_player = UINT8_MAX;
    lane->observation.followers[slot].source_player = UINT8_MAX;
  }
  for (int slot = 0; slot < 2; ++slot) {
    int roster = 1 - slot;
    MslObservationPlayer* player = &lane->observation.slots[slot];
    player->source_player = (uint8_t)roster;
    player->present = 1;
    player->stocks = 4;
    player->jumps_left = 1;
    player->hitlag_left = 1.25f;
    player->shield_hp = 60.0f;
    player->pos_x = 10.0f + (float)roster;
    player->facing = (uint8_t)roster;
  }
  if (config->players[1].character == MSL_CHARACTER_ICE_CLIMBERS) {
    MslObservationPlayer* nana = &lane->observation.followers[0];
    nana->present = 1;
    nana->source_player = 1;
    nana->pos_x = -9.0f;
    nana->stocks = 4;
    nana->jumps_left = 2;
  }
  int spawn_ids[] = {7, 3, 12, 5, 1};
  for (int item = 0; item < 5; ++item) {
    MslItem* output = &lane->observation.items[item];
    output->exists = 1;
    output->spawn_id = (uint32_t)spawn_ids[item];
    output->type = (uint16_t)spawn_ids[item];
  }
  RewardOverride* override = &reward_override[index];
  if (override->enabled) {
    for (int port = 0; port < 2; ++port) {
      lane->observation.slots[port].stocks = override->old_stock[port];
      lane->observation.slots[port].percent = override->old_percent[port];
    }
  }
  if (bad_roster_on_reset[index]) {
    lane->observation.slots[1].source_player = UINT8_MAX;
    bad_roster_on_reset[index] = 0;
  }
}

MslResult msl_batch_reset(MslBatch* batch, const MslMatchConfig configs[],
                          const uint8_t reset_mask[], MslObservation observations[]) {
  if (!batch || !configs || !observations) return MSL_INVALID_ARGUMENT;
  for (uint32_t index = 0; index < batch->size; ++index) {
    if (!reset_mask || reset_mask[index]) reset_lane(&batch->lanes[index], &configs[index], index);
    observations[index] = batch->lanes[index].observation;
  }
  return MSL_OK;
}

MslResult msl_batch_observe(const MslBatch* batch, MslObservation observations[],
                            MslTerminal terminals[]) {
  if (!batch || !observations || !terminals) return MSL_INVALID_ARGUMENT;
  for (uint32_t index = 0; index < batch->size; ++index) {
    if (!batch->lanes[index].active) return MSL_INVALID_STATE;
  }
  for (uint32_t index = 0; index < batch->size; ++index) {
    observations[index] = batch->lanes[index].observation;
    terminals[index] = batch->lanes[index].terminal;
  }
  return MSL_OK;
}

MslResult msl_batch_step_masked(MslBatch* batch, const MslInput inputs[],
                                const uint8_t step_mask[],
                                MslObservation observations[], MslTerminal terminals[]) {
  if (!batch || !inputs || !observations || !terminals) return MSL_INVALID_ARGUMENT;
  for (uint32_t index = 0; index < batch->size; ++index) {
    if (!batch->lanes[index].active) return MSL_INVALID_STATE;
  }
  for (uint32_t index = 0; index < batch->size; ++index) {
    TestLane* lane = &batch->lanes[index];
    if (!step_mask || step_mask[index]) {
      if (!lane->active) return MSL_INVALID_STATE;
      last_input[index] = inputs[index];
      lane->observation.frame_id += 1;
      lane->steps += 1;
      memset(&lane->terminal, 0, sizeof(lane->terminal));
      lane->terminal.frame_id = lane->observation.frame_id;
      lane->terminal.stage_id = lane->observation.stage_id;
      RewardOverride* override = &reward_override[index];
      if (override->enabled) {
        for (int port = 0; port < 2; ++port) {
          lane->observation.slots[port].stocks = override->new_stock[port];
          lane->observation.slots[port].percent = override->new_percent[port];
        }
      } else if (lane->config.max_frame == -1 && lane->steps == 1) {
        lane->observation.slots[1].stocks = 3;
        lane->observation.slots[0].percent = 0.3f;
        lane->observation.slots[1].percent = 10.1f;
      } else if (lane->config.max_frame == -1 && lane->steps == 2) {
        lane->observation.slots[1].stocks = 0;
        lane->terminal.stockout = 1;
      }
      if (disappear_on_step[index]) {
        lane->observation.slots[0].present = 0;
        lane->observation.slots[0].stocks = 3;
        lane->observation.followers[0].present = 0;
        for (int item = 0; item < MSL_MAX_ITEMS; ++item) {
          lane->observation.items[item].exists = 0;
        }
        lane->observation.items[4].exists = 1;
        disappear_on_step[index] = 0;
      }
      if (lane->config.max_frame >= 0 &&
          lane->observation.frame_id >= lane->config.max_frame) {
        lane->terminal.max_frame_reached = 1;
      }
    }
    observations[index] = lane->observation;
    terminals[index] = lane->terminal;
  }
  return MSL_OK;
}

MslResult msl_batch_step(MslBatch* batch, const MslInput inputs[],
                         MslObservation observations[], MslTerminal terminals[]) {
  return msl_batch_step_masked(batch, inputs, NULL, observations, terminals);
}

MslResult msl_batch_copy(MslBatch* destination, const MslBatch* source,
                         const uint32_t destination_indices[],
                         const uint32_t source_indices[], uint32_t count) {
  if (!destination || !source || !destination_indices || !source_indices) {
    return MSL_INVALID_ARGUMENT;
  }
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t dst = destination_indices[index];
    uint32_t src = source_indices[index];
    if (dst >= destination->size || src >= source->size) return MSL_INVALID_ARGUMENT;
    destination->lanes[dst] = source->lanes[src];
  }
  return MSL_OK;
}

MslResult msl_batch_save_size(const MslBatch* batch, uint32_t index,
                              size_t* required_size) {
  if (!batch || index >= batch->size || !required_size) return MSL_INVALID_ARGUMENT;
  *required_size = sizeof(TestLane);
  return MSL_OK;
}

MslResult msl_batch_save(const MslBatch* batch, uint32_t index, void* buffer,
                         size_t buffer_size, size_t* written) {
  if (!batch || index >= batch->size || !buffer || !written ||
      buffer_size < sizeof(TestLane)) return MSL_INVALID_ARGUMENT;
  memcpy(buffer, &batch->lanes[index], sizeof(TestLane));
  *written = sizeof(TestLane);
  return MSL_OK;
}

MslResult msl_batch_restore(MslBatch* batch, uint32_t index, const void* buffer,
                            size_t buffer_size) {
  if (!batch || index >= batch->size || !buffer ||
      buffer_size != sizeof(TestLane)) return MSL_INVALID_ARGUMENT;
  memcpy(&batch->lanes[index], buffer, sizeof(TestLane));
  return MSL_OK;
}
