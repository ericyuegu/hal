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

// Recorded playback uses the same packed representation as the replay validator.
// Config.frame_id is the state BEFORE the first replay tick (normally -124).
// Unlike msl_batch_reset, reset_replay does not perform a hidden neutral tick.
#pragma pack(push, 1)
typedef struct MslReplayInputPlayer {
    uint16_t buttons;
    // Physical signed bytes retained for UCF's raw-input consumers.
    int8_t main_x;
    int8_t main_y;
    int8_t c_x;
    int8_t c_y;
    uint8_t l;
    uint8_t r;
    // Optional post-clamp HSD axes. Slippi playback restores these floats to
    // Fighter input state independently of the physical bytes above.
    int8_t nml_main_x;
    int8_t nml_main_y;
    int8_t nml_c_x;
    int8_t nml_c_y;
    uint8_t nml_valid;
} MslReplayInputPlayer;

enum {
    MSL_REPLAY_INPUT_NML_MAIN_VALID = 1 << 0,
    MSL_REPLAY_INPUT_NML_C_VALID = 1 << 1,
};

typedef struct MslReplayInput {
    MslReplayInputPlayer p[MSL_MAX_PLAYERS];
} MslReplayInput;

// Slippi Playback/Core/RestoreGameFrame.asm publishes processed inputs at
// 0x8006B0DC, independently of physical controller bytes. Validation also
// carries the RNG observation from Recording/SendGamePreFrame.asm.
typedef struct MslReplayCpuInput {
    float main_x, main_y, c_x, c_y, trigger;
    uint32_t buttons;
    uint32_t random_seed;
    uint8_t valid;
} MslReplayCpuInput;

typedef struct MslReplayStageEvents {
    float fod_platform_height[2];
    // Slippi records RNG at frame start and at the first fighter input
    // callback. Stage and effect callbacks can consume RNG between those points.
    uint32_t fighter_pre_random_seed;
    uint8_t fod_platform_mask;
    uint8_t dreamland_whispy_valid;
    uint8_t dreamland_whispy_direction;
    uint8_t fighter_pre_random_seed_valid;
    MslReplayCpuInput cpu_inputs[MSL_MAX_PLAYERS];
} MslReplayStageEvents;

typedef struct MslReplayFrame {
    uint32_t frame_pre_random_seed;
    MslReplayInput input;
    MslReplayStageEvents stage_events;
} MslReplayFrame;

enum { MSL_FACING_AUTO_FLAG = 0x10 };

typedef struct MslReplayPlayerConfig {
    uint8_t char_id;
    uint8_t team_id;
    // Bit 0 is facing. Bits 1..3 optionally carry the one-based physical
    // controller port; zero selects the source index and automatic facing.
    // Bit 4 selects automatic facing with an explicit controller port.
    uint8_t facing_and_port;
    uint8_t costume_id;
    uint8_t handicap;
    uint8_t start_percent;
    uint8_t cpu_level; // Zero selects a controller; 1..9 retain a CPU slot.
} MslReplayPlayerConfig;

typedef struct MslReplayConfig {
    uint32_t stage_id;
    int32_t frame_id;
    uint32_t frame_pre_random_seed;
    // Slippi's game-start event records the seed restored before match/stage
    // construction. Online frame seeds subsequently add a per-frame high-word
    // offset and must not initialize persistent stage state.
    uint32_t initial_random_seed;
    float match_damage_ratio;
    uint8_t num_players;
    uint8_t is_teams;
    uint8_t friendly_fire;
    uint8_t stock_count;
    uint8_t camera_mode;
    // Runtime capabilities are explicit because scene major alone does not
    // distinguish Slippi's offline Dolphin patch set from Nintendont retail.
    uint8_t online_fnmsubs_zero;
    uint8_t brawl_offscreen_damage;
    uint8_t freeze_dead_up_fall_physics;
    // UCF 0.84 and its 1.0 cardinal-coordinate patch are independently
    // selectable in real Slippi recordings. Older recordings may contain
    // UCF dashback/shield-drop behavior without cardinal snapping.
    // refs/slippi-ssbm-asm/Output/InjectionLists/list_netplay.json
    uint8_t ucf_cardinals_1_0_enabled;
    // UCF's shield-SDI injection was not present in every Slippi UCF profile.
    // Keep it independent of dashback, shield drop, and cardinal snapping so
    // older recordings do not receive a second, unrecorded SDI displacement.
    // refs/slippi-ssbm-asm/External/UCF 0.84/UCF/UCF Shield SDI.asm
    uint8_t ucf_shield_sdi_enabled;
    // UCF's ordinary hitlag-SDI injection is a separate rollout/capability
    // from shield SDI and from the original dashback/shield-drop mechanics.
    // refs/slippi-ssbm-asm/External/UCF 0.84/UCF/UCF SDI.asm
    uint8_t ucf_sdi_enabled;
    // The counter-based (extended) shield drop is likewise its own rollout;
    // recordings that predate it keep the classic shield-drop stick check
    // only.
    // refs/slippi-ssbm-asm/External/UCF 0.84/UCF/UCF Shield Drop Extended.asm
    uint8_t ucf_shield_drop_extended_enabled;
    // The classic shield drop itself shipped in two versions. UCF 0.84 gates
    // the spot-dodge suppression on standing on a platform line; the earlier
    // UCF 0.8 code (unchanged since 0.73) has no platform check and applies
    // its float rim test everywhere, so a rim-coordinate shield press with the
    // stick held down on the main floor shields instead of spot-dodging.
    // Recordings whose embedded gecko list carries the 0.8 code clear this.
    // refs/slippi-ssbm-asm/External/UCF 0.84/UCF/UCF Shield Drop.asm
    // refs/slippi-ssbm-asm/External/UCF 0.8/Logic/UCF SD.asm
    uint8_t ucf_shield_drop_084_enabled;
    // Bit 0: Slippi 3.18+ FoD platform events; bit 1: Dream Land Whispy
    // direction events. A present stream is authoritative even on frames with
    // no event, where the last published value remains active.
    uint8_t stage_event_streams;
    // Online/Core/WhispyBlowDirFix excludes death animations from the vote.
    uint8_t whispy_dead_fighter_fix;
    MslReplayPlayerConfig players[MSL_MAX_PLAYERS];
} MslReplayConfig;

#pragma pack(pop)

MSL_API const char* msl_result_string(MslResult result);
// Increment when a public C struct layout or field meaning changes.
MSL_API uint32_t msl_api_abi_version(void);
// Replay layouts and semantics have their own identity; currently version 1.
MSL_API uint32_t msl_replay_abi_version(void);
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

// Replay entry points preserve recorded arithmetic/UCF capabilities, both RNG
// boundaries, processed inputs, and stage events. Masks select lanes as above.
MSL_API MslResult msl_batch_reset_replay(MslBatch* batch, const MslReplayConfig configs[],
                                       const MslReplayInput previous[], const uint8_t mask[]);
MSL_API MslResult msl_batch_step_replay(MslBatch* batch, const MslReplayFrame frames[],
                                      const uint8_t mask[], MslObservation observations[],
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
