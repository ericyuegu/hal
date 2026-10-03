#ifndef MSL_API_H
#define MSL_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MSL_CORE_SHARED) && defined(__GNUC__)
#define MSL_API __attribute__((visibility("default")))
#else
#define MSL_API
#endif

enum {
  MSL_MAX_PLAYERS = 4,
  MSL_MAX_ITEMS = 15,
};

typedef struct MslBatch MslBatch;

typedef enum MslResult {
  MSL_OK = 0,
  MSL_INVALID_ARGUMENT = 1,
  MSL_OUT_OF_MEMORY = 2,
  MSL_INVALID_STATE = 3,
  MSL_INCOMPATIBLE = 4,
} MslResult;

typedef enum MslCharacter {
  MSL_CHARACTER_MARIO = 0,
  MSL_CHARACTER_FOX = 1,
  MSL_CHARACTER_MEWTWO = 16,
  MSL_CHARACTER_GAMEWATCH = 24,
  MSL_CHARACTER_CAPTAIN_FALCON = 2,
  MSL_CHARACTER_DONKEY_KONG = 3,
  MSL_CHARACTER_GANONDORF = 25,
  MSL_CHARACTER_YOSHI = 14,
  MSL_CHARACTER_BOWSER = 5,
  MSL_CHARACTER_SHEIK = 7,
  MSL_CHARACTER_PEACH = 9,
  MSL_CHARACTER_ICE_CLIMBERS = 10,
  MSL_CHARACTER_PIKACHU = 12,
  MSL_CHARACTER_PICHU = 23,
  MSL_CHARACTER_KIRBY = 4,
  MSL_CHARACTER_SAMUS = 13,
  MSL_CHARACTER_NESS = 8,
  MSL_CHARACTER_LINK = 6,
  MSL_CHARACTER_YOUNG_LINK = 20,
  MSL_CHARACTER_JIGGLYPUFF = 15,
  MSL_CHARACTER_LUIGI = 17,
  MSL_CHARACTER_DRMARIO = 21,
  MSL_CHARACTER_MARTH = 18,
  MSL_CHARACTER_ROY = 26,
  MSL_CHARACTER_ZELDA = 19,
  MSL_CHARACTER_FALCO = 22,
} MslCharacter;

typedef enum MslStage {
  MSL_STAGE_FOUNTAIN_OF_DREAMS = 2,
  MSL_STAGE_POKEMON_STADIUM = 3,
  MSL_STAGE_YOSHIS_STORY = 8,
  MSL_STAGE_DREAM_LAND_N64 = 28,
  MSL_STAGE_BATTLEFIELD = 31,
  MSL_STAGE_FINAL_DESTINATION = 32,
} MslStage;

typedef enum MslWhispyBlowDirection {
  MSL_WHISPY_NONE = 0,
  MSL_WHISPY_LEFT = 1,
  MSL_WHISPY_RIGHT = 2,
} MslWhispyBlowDirection;

typedef enum MslButton {
  MSL_BUTTON_D_DOWN = 0x0004,
  MSL_BUTTON_D_UP = 0x0008,
  MSL_BUTTON_Z = 0x0010,
  MSL_BUTTON_R = 0x0020,
  MSL_BUTTON_L = 0x0040,
  MSL_BUTTON_A = 0x0100,
  MSL_BUTTON_B = 0x0200,
  MSL_BUTTON_X = 0x0400,
  MSL_BUTTON_Y = 0x0800,
  MSL_BUTTON_START = 0x1000,
} MslButton;

typedef enum MslFacing {
  MSL_FACING_LEFT = -1,
  MSL_FACING_AUTO = 0,
  MSL_FACING_RIGHT = 1,
} MslFacing;

typedef enum MslTeamRelation {
  MSL_TEAM_SELF = 0,
  MSL_TEAM_ALLY = 1,
  MSL_TEAM_OPPONENT = 2,
} MslTeamRelation;

enum {
  MSL_TEAM_AUTO = -1,
  MSL_CONTROLLER_PORT_AUTO = -1,
};

typedef struct MslPlayerConfig {
  uint8_t character;
  int8_t team;
  int8_t facing;
  int8_t controller_port;
  uint8_t costume;
  uint8_t handicap;
  uint8_t start_percent; /* Initial damage, 0..100; zero on later stock respawns. */
} MslPlayerConfig;

typedef struct MslMatchConfig {
  uint32_t stage;
  uint32_t random_seed;
  int32_t max_frame;
  float damage_ratio;
  uint8_t num_players;
  uint8_t is_teams;
  uint8_t friendly_fire;
  uint8_t stocks;
  uint8_t viewpoint_player;
  uint8_t ucf_cardinals;
  MslPlayerConfig players[MSL_MAX_PLAYERS];
} MslMatchConfig;

typedef struct MslInputPlayer {
  uint16_t buttons;
  int8_t main_x;
  int8_t main_y;
  int8_t c_x;
  int8_t c_y;
  uint8_t l;
  uint8_t r;
} MslInputPlayer;

typedef struct MslInput {
  MslInputPlayer players[MSL_MAX_PLAYERS];
} MslInput;

typedef struct MslItem {
  uint8_t exists;
  uint8_t state;
  uint16_t type;
  int8_t owner;
  uint8_t _pad0;
  uint16_t instance_id;
  uint16_t attack_id;
  uint16_t attack_instance;
  float direction;
  float vel_x;
  float vel_y;
  float pos_x;
  float pos_y;
  uint16_t damage;
  uint16_t _pad1;
  float timer;
  uint32_t spawn_id;
  uint8_t misc0;
  uint8_t misc1;
  uint8_t misc2;
  uint8_t misc3;
} MslItem;

typedef struct MslObservationPlayer {
  uint8_t present; /* Fighter is awake; roster metadata remains when absent. */
  uint8_t source_player; /* UINT8_MAX for unused roster slots. */
  uint8_t team_relation;
  uint8_t team_id;
  float pos_x;
  float pos_y;
  float speed_air_x_self;
  float speed_ground_x_self;
  float speed_y_self;
  float speed_x_attack;
  float speed_y_attack;
  float percent;
  float shield_hp;
  uint16_t action_id;
  int16_t action_frame;
  uint16_t hitlag;
  uint16_t hitstun;
  uint8_t char_id;
  uint8_t stocks;
  uint8_t facing;
  uint8_t on_ground;
  uint8_t jumps_left;
  uint8_t hurtbox_state;
  uint8_t invulnerable;
  uint8_t _pad0;
  float hitlag_left;
} MslObservationPlayer;

typedef struct MslObservationRandall {
  uint8_t exists;
  uint8_t _pad0[3];
  float x;
  float y;
} MslObservationRandall;

typedef struct MslObservationFodPlatforms {
  float left;
  float right;
} MslObservationFodPlatforms;

typedef struct MslObservationStage {
  MslObservationRandall randall;
  MslObservationFodPlatforms fod_platforms;
  uint8_t whispy; // MSL_WHISPY_*; NONE outside Dream Land.
  uint8_t _pad0[3];
} MslObservationStage;

typedef struct MslObservation {
  int32_t frame_id;
  uint32_t frame_pre_random_seed;
  uint32_t stage_id;
  uint8_t num_players;
  uint8_t viewpoint_player;
  uint8_t is_teams;
  uint8_t _pad0;
  MslObservationStage stage;
  MslObservationPlayer slots[MSL_MAX_PLAYERS];
  MslItem items[MSL_MAX_ITEMS];
  // Ice Climbers' Nana: followers[k] is the follower of the player in
  // slots[k], present only while Slippi would record a follower row for her
  // (awake, including her death animation; not while asleep before Rebirth).
  MslObservationPlayer followers[MSL_MAX_PLAYERS];
} MslObservation;

typedef struct MslTerminal {
  int32_t frame_id;
  uint32_t stage_id;
  uint8_t done;
  uint8_t match_ended;
  uint8_t stockout;
  uint8_t max_frame_reached;
  uint8_t alive_count;
  uint8_t alive_team_count;
  uint8_t team_alive_mask;
  uint8_t _pad0;
} MslTerminal;

MSL_API const char* msl_result_string(MslResult result);
// Increment when a public C struct layout or field meaning changes.
MSL_API uint32_t msl_api_abi_version(void);
MSL_API MslMatchConfig msl_match_config_default(void);

// Immutable game data is loaded once per process, on the first
// msl_batch_create or msl_game_data_acquire, and shared by every batch; a
// process holds a single data root. Acquire it ahead of time to pay the load
// once for processes forked afterwards, which then share its pages
// copy-on-write. Each acquire is balanced by a release; the data is freed
// when no batch or acquire holds it.
MSL_API MslResult msl_game_data_acquire(const char* data_root);
MSL_API void msl_game_data_release(void);
MSL_API uint32_t msl_game_data_references(void);

// data_root is the extraction root or its raw/ directory.
MSL_API MslResult msl_batch_create(const char* data_root, uint32_t batch_size,
                                   MslBatch** out_batch);
MSL_API void msl_batch_destroy(MslBatch* batch);
MSL_API uint32_t msl_batch_size(const MslBatch* batch);

// A NULL reset mask resets every environment; otherwise nonzero bytes select.
MSL_API MslResult msl_batch_reset(MslBatch* batch, const MslMatchConfig configs[],
                                  const uint8_t reset_mask[], MslObservation observations[]);
MSL_API MslResult msl_batch_step(MslBatch* batch, const MslInput inputs[],
                                 MslObservation observations[], MslTerminal terminals[]);
// A NULL step mask steps every environment; otherwise nonzero bytes select.
// Every environment is observed regardless of the mask, so unstepped matches
// republish their current state (e.g. a fresh post-reset entry frame).
MSL_API MslResult msl_batch_step_masked(MslBatch* batch, const MslInput inputs[],
                                        const uint8_t step_mask[],
                                        MslObservation observations[], MslTerminal terminals[]);
MSL_API MslResult msl_batch_observe(const MslBatch* batch, MslObservation observations[],
                                    MslTerminal terminals[]);

MSL_API MslResult msl_batch_copy(MslBatch* destination, const MslBatch* source,
                                 const uint32_t destination_indices[],
                                 const uint32_t source_indices[], uint32_t count);
MSL_API MslResult msl_batch_save_size(const MslBatch* batch, uint32_t env_index,
                                      size_t* required_size);
MSL_API MslResult msl_batch_save(const MslBatch* batch, uint32_t env_index, void* buffer,
                                 size_t buffer_size, size_t* written);
MSL_API MslResult msl_batch_restore(MslBatch* batch, uint32_t env_index, const void* buffer,
                                    size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
