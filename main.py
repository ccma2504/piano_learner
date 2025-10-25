import pygame
import mido
import sys
import numpy as np
import sounddevice as sd
import threading
import os
import scipy.io.wavfile
import scipy.signal

# --- Pygame and Font Initialization ---
pygame.init()
pygame.font.init()

# --- Constants and Basic Settings ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
HIGHLIGHT_COLOR = (173, 216, 230)
MIDI_NOTE_COLOR = (144, 238, 144)
MENU_COLOR = (60, 60, 60)
MENU_SELECTED_COLOR = (90, 90, 90)

TOTAL_KEYS = 88
WHITE_KEYS_COUNT = 52
WHITE_KEY_WIDTH = SCREEN_WIDTH / WHITE_KEYS_COUNT
WHITE_KEY_HEIGHT = 200
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH * 0.6
BLACK_KEY_HEIGHT = WHITE_KEY_HEIGHT * 0.6

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- Global Variables ---
actual_sample_rate = 0
active_audio_notes = set()
note_playback_positions = {}
audio_lock = threading.Lock()

midi_notes = []
NOTE_FALL_SPEED = 200
playback_rate = 1.0

midi_audio_enabled = True
user_audio_enabled = True
paused = False

app_state = 'MENU'
midi_file_list = []
selected_midi_index = 0

loop_start_time = None
loop_end_time = None

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)

piano_keys = []
piano_key_map = {}
loaded_samples = {}

# --- Function Definitions ---

def get_midi_files():
    global midi_file_list
    midi_dir = "midi"
    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)
    midi_file_list = [f for f in os.listdir(midi_dir) if f.lower().endswith(('.mid', '.midi'))]
    return midi_file_list

def draw_menu():
    screen.fill(GRAY)
    title_text = font.render("Select a MIDI File (↑/↓, Enter)", True, BLACK)
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH / 2, 50))
    screen.blit(title_text, title_rect)

    if not midi_file_list:
        error_text = font.render("Please put MIDI files in the 'midi' folder.", True, BLACK)
        error_rect = error_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(error_text, error_rect)
        return

    for i, filename in enumerate(midi_file_list):
        color = MENU_COLOR
        if i == selected_midi_index:
            color = MENU_SELECTED_COLOR
        
        rect = pygame.Rect(SCREEN_WIDTH / 4, 100 + i * 50, SCREEN_WIDTH / 2, 40)
        pygame.draw.rect(screen, color, rect)
        
        file_text = font_small.render(filename, True, WHITE)
        text_rect = file_text.get_rect(center=rect.center)
        screen.blit(file_text, text_rect)

def build_piano_keys():
    white_key_x = 0
    white_keys_rects = []
    for i in range(TOTAL_KEYS):
        midi_note = 21 + i
        is_b = (midi_note % 12) in [1, 3, 6, 8, 10]
        if not is_b:
            rect = pygame.Rect(white_key_x, SCREEN_HEIGHT - WHITE_KEY_HEIGHT, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT)
            key_info = {'rect': rect, 'midi_note': midi_note, 'type': 'white', 'color': WHITE}
            piano_keys.append(key_info)
            piano_key_map[midi_note] = key_info
            white_keys_rects.append(rect)
            white_key_x += WHITE_KEY_WIDTH
    white_key_index = 0
    for i in range(TOTAL_KEYS):
        midi_note = 21 + i
        is_b = (midi_note % 12) in [1, 3, 6, 8, 10]
        if not is_b: white_key_index += 1
        else:
            prev_white_key_rect = white_keys_rects[white_key_index - 1]
            rect = pygame.Rect(prev_white_key_rect.right - BLACK_KEY_WIDTH / 2, SCREEN_HEIGHT - WHITE_KEY_HEIGHT, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)
            key_info = {'rect': rect, 'midi_note': midi_note, 'type': 'black', 'color': BLACK}
            piano_keys.append(key_info)
            piano_key_map[midi_note] = key_info

def draw_piano(surface, user_notes, midi_notes_to_hit):
    for key in piano_keys:
        if key['type'] == 'white':
            color = key['color']
            if key['midi_note'] in midi_notes_to_hit: color = MIDI_NOTE_COLOR
            if key['midi_note'] in user_notes: color = HIGHLIGHT_COLOR
            pygame.draw.rect(surface, color, key['rect'])
            pygame.draw.rect(surface, BLACK, key['rect'], 1)
    for key in piano_keys:
        if key['type'] == 'black':
            color = key['color']
            if key['midi_note'] in midi_notes_to_hit: color = MIDI_NOTE_COLOR
            if key['midi_note'] in user_notes: color = HIGHLIGHT_COLOR
            pygame.draw.rect(surface, color, key['rect'])

def load_midi_file(filepath):
    global midi_notes, loop_start_time, loop_end_time
    loop_start_time, loop_end_time = None, None
    midi_notes = []
    try:
        mid = mido.MidiFile(filepath)
        current_time = 0
        active_notes = {}
        for msg in mid:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {'start': current_time, 'velocity': msg.velocity}
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_info = active_notes.pop(msg.note)
                    midi_notes.append({
                        'note': msg.note,
                        'start_time': start_info['start'],
                        'end_time': current_time,
                        'triggered_on': False,
                        'triggered_off': False
                    })
        print(f"Loaded {len(midi_notes)} notes. (Total length: {current_time:.2f}s)")
        pygame.display.set_caption("Piano Learner - " + os.path.basename(filepath))
    except Exception as e:
        print(f"Error loading MIDI file: {e}", file=sys.stderr)
        return False
    return True

def draw_falling_notes(surface, current_playback_time):
    if NOTE_FALL_SPEED <= 0: return
    visible_time_window = SCREEN_HEIGHT / NOTE_FALL_SPEED
    for note_info in midi_notes:
        start_time = note_info['start_time']
        end_time = note_info['end_time']
        note_midi = note_info['note']
        if start_time > current_playback_time and start_time < current_playback_time + visible_time_window:
            if note_midi in piano_key_map:
                key_info = piano_key_map[note_midi]
                key_rect = key_info['rect']
                time_until_hit = start_time - current_playback_time
                y_pos = (SCREEN_HEIGHT - WHITE_KEY_HEIGHT) - (time_until_hit * (NOTE_FALL_SPEED / playback_rate))
                note_duration = end_time - start_time
                note_height = max(1, note_duration * (NOTE_FALL_SPEED / playback_rate))
                note_color = MIDI_NOTE_COLOR
                if key_info['type'] == 'black': note_color = (50, 150, 50)
                falling_note_rect = pygame.Rect(key_rect.x, y_pos - note_height, key_rect.width, note_height)
                pygame.draw.rect(surface, note_color, falling_note_rect)
                pygame.draw.rect(surface, BLACK, falling_note_rect, 1)
                if falling_note_rect.height > 15:
                    note_name = NOTE_NAMES[note_midi % 12]
                    note_text = font_small.render(note_name, True, BLACK)
                    text_rect = note_text.get_rect(center=falling_note_rect.center)
                    surface.blit(note_text, text_rect)

def load_piano_samples_jobro():
    global actual_sample_rate, loaded_samples
    SAMPLE_DIR = "samples/2489__jobro__piano-ff/"
    print(f"Loading piano samples from ({SAMPLE_DIR})...")
    midi_offset = 36
    try:
        for i in range(1, 89):
            filename = f"{i:03d}.wav"
            filepath = os.path.join(SAMPLE_DIR, filename)
            midi_note = midi_offset + (i - 1)
            if os.path.exists(filepath):
                samplerate, data = scipy.io.wavfile.read(filepath)
                if actual_sample_rate == 0:
                    actual_sample_rate = samplerate
                    print(f"Reference sample rate set: {actual_sample_rate} Hz")
                if data.dtype == np.int16: data = data.astype(np.float32) / 32768.0
                elif data.dtype != np.float32: data = data.astype(np.float32)
                if data.ndim == 1: data = np.column_stack((data, data))
                loaded_samples[midi_note] = data
    except Exception as e:
        print(f"Critical error loading samples: {e}", file=sys.stderr)
    print(f"Loaded {len(loaded_samples)} piano samples.")

def audio_callback(outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    outdata.fill(0)
    with audio_lock:
        notes_to_remove = set()
        for note in list(active_audio_notes):
            sample = loaded_samples.get(note + 12)
            if sample is None:
                notes_to_remove.add(note)
                continue
            pos = note_playback_positions.get(note, 0)
            remaining_in_sample = len(sample) - pos
            frames_to_write = min(frames, remaining_in_sample)
            if frames_to_write > 0:
                chunk = sample[pos : pos + frames_to_write]
                outdata[:frames_to_write] += chunk * 0.5
                note_playback_positions[note] = pos + frames_to_write
        for note in notes_to_remove:
            if note in active_audio_notes: active_audio_notes.remove(note)
            if note in note_playback_positions: del note_playback_positions[note]
    np.clip(outdata, -1.0, 1.0, out=outdata)

def restart_playback():
    global current_playback_time_sec, loop_start_time, loop_end_time
    current_playback_time_sec = 0.0
    loop_start_time, loop_end_time = None, None
    with audio_lock:
        active_audio_notes.clear()
        note_playback_positions.clear()
    for i in range(len(midi_notes)):
        midi_notes[i]['triggered_on'] = False
        midi_notes[i]['triggered_off'] = False
    print("Playback restarted.")

# --- Initialization ---
build_piano_keys()
get_midi_files()
load_piano_samples_jobro()

midi_port = None
try:
    port_names = [name for name in mido.get_input_names() if "iCON iKeyboard" in name]
    if port_names:
        port_name = port_names[1] if len(port_names) > 1 else port_names[0]
        midi_port = mido.open_input(port_name)
        print(f"Success: Connected to MIDI port '{port_name}'.")
except Exception as e:
    print(f"Warning: Could not find MIDI device. ({e})")

if actual_sample_rate == 0:
    print("Fatal: Could not load audio samples. Exiting.", file=sys.stderr)
    pygame.quit()
    sys.exit()

stream = sd.OutputStream(channels=2, callback=audio_callback, samplerate=actual_sample_rate, dtype='float32')
stream.start()
print("Low-latency audio stream started.")

# --- Main Loop ---
running = True
clock = pygame.time.Clock()
active_visual_notes = set()
current_playback_time_sec = 0.0

while running:
    if app_state == 'MENU':
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_DOWN:
                    if midi_file_list: selected_midi_index = (selected_midi_index + 1) % len(midi_file_list)
                elif event.key == pygame.K_UP:
                    if midi_file_list: selected_midi_index = (selected_midi_index - 1) % len(midi_file_list)
                elif event.key == pygame.K_RETURN:
                    if midi_file_list:
                        filepath = os.path.join("midi", midi_file_list[selected_midi_index])
                        if load_midi_file(filepath):
                            restart_playback()
                            app_state = 'PLAYING'
        draw_menu()

    elif app_state == 'PLAYING':
        dt = clock.tick(120) / 1000.0
        if not paused:
            if loop_start_time is not None and loop_end_time is not None and current_playback_time_sec >= loop_end_time:
                current_playback_time_sec = loop_start_time
                for i in range(len(midi_notes)):
                    if loop_start_time <= midi_notes[i]['start_time'] < loop_end_time:
                        midi_notes[i]['triggered_on'] = False
                        midi_notes[i]['triggered_off'] = False
            else:
                current_playback_time_sec += dt * playback_rate

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    app_state = 'MENU'
                    with audio_lock:
                        active_audio_notes.clear()
                        note_playback_positions.clear()
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_r: restart_playback()
                elif event.key == pygame.K_m: midi_audio_enabled = not midi_audio_enabled
                elif event.key == pygame.K_u: user_audio_enabled = not user_audio_enabled
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    playback_rate += 0.1
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    playback_rate = max(0.1, playback_rate - 0.1)
                elif event.key == pygame.K_F1: 
                    loop_start_time = current_playback_time_sec
                    loop_end_time = None
                elif event.key == pygame.K_F2:
                    if loop_start_time is not None and current_playback_time_sec > loop_start_time:
                        loop_end_time = current_playback_time_sec
                elif event.key == pygame.K_F3: 
                    loop_start_time, loop_end_time = None, None

        if not paused and midi_port:
            for msg in midi_port.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_visual_notes.add(msg.note)
                    if user_audio_enabled:
                        with audio_lock:
                            active_audio_notes.add(msg.note)
                            note_playback_positions[msg.note] = 0
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_visual_notes: active_visual_notes.remove(msg.note)
                    if user_audio_enabled:
                        with audio_lock:
                            if msg.note in active_audio_notes: active_audio_notes.remove(msg.note)
                            if msg.note in note_playback_positions: del note_playback_positions[msg.note]

        if not paused:
            midi_notes_to_hit = set()
            with audio_lock:
                user_held_notes = active_visual_notes if user_audio_enabled else set()
                for i, note_info in enumerate(midi_notes):
                    note = note_info['note']
                    if not note_info['triggered_on'] and note_info['start_time'] <= current_playback_time_sec:
                        if midi_audio_enabled:
                            active_audio_notes.add(note)
                            note_playback_positions[note] = 0
                        midi_notes[i]['triggered_on'] = True
                    if not note_info['triggered_off'] and note_info['end_time'] <= current_playback_time_sec:
                        if note in active_audio_notes and note not in user_held_notes:
                            active_audio_notes.remove(note)
                            if note in note_playback_positions: del note_playback_positions[note]
                        midi_notes[i]['triggered_off'] = True
                    if note_info['start_time'] <= current_playback_time_sec < note_info['end_time']:
                         midi_notes_to_hit.add(note)

        screen.fill(GRAY)
        draw_falling_notes(screen, current_playback_time_sec)
        draw_piano(screen, active_visual_notes, midi_notes_to_hit)

        info_y = 10
        speed_text = font_small.render(f"Playback: {playback_rate:.1f}x (+/-)", True, BLACK)
        screen.blit(speed_text, (10, info_y)); info_y += 25
        midi_audio_text = font_small.render(f"MIDI Audio (M): {'ON' if midi_audio_enabled else 'OFF'}", True, BLACK)
        screen.blit(midi_audio_text, (10, info_y)); info_y += 25
        user_audio_text = font_small.render(f"User Audio (U): {'ON' if user_audio_enabled else 'OFF'}", True, BLACK)
        screen.blit(user_audio_text, (10, info_y)); info_y += 25
        controls_text_1 = font_small.render("Space: Pause, R: Restart, ESC: Menu", True, BLACK)
        screen.blit(controls_text_1, (10, info_y)); info_y += 25
        controls_text_2 = font_small.render("F1: Loop Start, F2: Loop End, F3: Loop Clear", True, BLACK)
        screen.blit(controls_text_2, (10, info_y)); info_y += 25

        if loop_start_time is not None:
            loop_text = f"Loop: {loop_start_time:.1f}s - {loop_end_time:.1f}s" if loop_end_time is not None else f"Loop Start: {loop_start_time:.1f}s"
            loop_surf = font_small.render(loop_text, True, (200, 0, 0))
            screen.blit(loop_surf, (10, info_y))

        if paused:
            pause_text = font.render("[ PAUSED ]", True, BLACK, WHITE)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            screen.blit(pause_text, pause_rect)

    pygame.display.flip()

# --- Cleanup ---
stream.stop()
stream.close()
if midi_port:
    midi_port.close()
pygame.quit()
sys.exit()