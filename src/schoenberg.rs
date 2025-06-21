use indexmap::IndexSet;
use midly::{
    num::{u28, u7},
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
};
use rand::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    error, fmt,
};

/// A struct representing a compiled Schoenberg program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// The tokens of the program, which are the instructions to be executed.
    tokens: Vec<Token>,
    /// A map from loop start and end `tokens` indices to loop end and start
    /// `tokens` indices, respectively.
    loop_boundary_indices: HashMap<usize, usize>,
}

impl Program {
    const MEMORY_SIZE: usize = 30_000;

    /// Runs the program with the given `input` and yields the output.
    pub fn run(&self, input: &str, output: impl Fn(u8)) {
        let input: &[u8] = input.as_bytes();
        let mut input_pointer = 0;

        let mut memory = [0u8; Self::MEMORY_SIZE];
        let mut memory_pointer = 0;
        let mut program_counter = 0;

        while program_counter < self.tokens.len() {
            let token = &self.tokens[program_counter];
            match token {
                Token::Decrement(amount) => {
                    memory[memory_pointer] = memory[memory_pointer].wrapping_sub(*amount)
                }
                Token::Increment(amount) => {
                    memory[memory_pointer] = memory[memory_pointer].wrapping_add(*amount)
                }
                Token::MoveLeft(amount) => {
                    memory_pointer =
                        memory_pointer.wrapping_sub((*amount).into()) % Self::MEMORY_SIZE
                }
                Token::MoveRight(amount) => {
                    memory_pointer =
                        memory_pointer.wrapping_add((*amount).into()) % Self::MEMORY_SIZE
                }
                Token::Output => output(memory[memory_pointer]),
                Token::Input => {
                    memory[memory_pointer] = match input.get(input_pointer) {
                        Some(&input) => {
                            input_pointer += 1;
                            input
                        }
                        None => 0,
                    }
                }
                Token::LoopStart => {
                    if memory[memory_pointer] == 0 {
                        program_counter = *self.loop_boundary_indices.get(&program_counter).unwrap()
                    }
                }
                Token::LoopEnd => {
                    if memory[memory_pointer] != 0 {
                        program_counter = *self.loop_boundary_indices.get(&program_counter).unwrap()
                    }
                }
            }
            program_counter += 1
        }
    }

    /// Compiles a [Program] from the given `midi_bytes` or returns a
    /// compilation error.
    pub fn from_midi(midi_bytes: &[u8]) -> Result<Self, Error> {
        let tokens: Vec<Token> = Self::tokenize(midi_bytes)?;
        let loop_boundary_indices = Self::find_loop_boundaries(&tokens);
        Ok(Program {
            tokens,
            loop_boundary_indices,
        })
    }

    /// Compiles a [Program] from the given `bf` program.
    ///
    /// Any non-BF characters are filtered out so it's always possible to
    /// compile a Schoenberg program.
    pub fn from_bf(bf: &str) -> Program {
        let mut tokens = Vec::new();

        let mut chars = bf
            .lines()
            .filter(|line| !line.starts_with("#"))
            .flat_map(|line| line.chars())
            .peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '+' | '-' | '>' | '<' => {
                    // Merge consecutive characters into a single token with the
                    // count.
                    let mut count = 1;
                    while chars.peek() == Some(&ch) {
                        chars.next();
                        count += 1;
                        if count == u8::MAX {
                            break;
                        }
                    }
                    tokens.push(match ch {
                        '+' => Token::Increment(count),
                        '-' => Token::Decrement(count),
                        '>' => Token::MoveRight(count),
                        '<' => Token::MoveLeft(count),
                        _ => unreachable!(),
                    });
                }
                '.' => tokens.push(Token::Output),
                ',' => tokens.push(Token::Input),
                '[' => tokens.push(Token::LoopStart),
                ']' => tokens.push(Token::LoopEnd),
                _ => {}
            }
        }

        let loop_boundary_indices = Self::find_loop_boundaries(&tokens);
        Self {
            tokens,
            loop_boundary_indices,
        }
    }

    /// Parses the given `midi_bytes` and extracts the tokens from the MIDI, or
    /// returns an error.
    fn tokenize(midi_bytes: &[u8]) -> Result<Vec<Token>, Error> {
        let smf = Smf::parse(midi_bytes).map_err(Error::Parse)?;
        if smf.header.format != midly::Format::SingleTrack {
            return Err(Error::MultipleTracks(smf.tracks.len()));
        }
        let track = smf.tracks.first().expect("Should have validated one track");

        let mut tokens = Vec::new();

        let mut keys_on: IndexSet<u7> = IndexSet::new();
        let mut last_key_on: Option<u7> = None;
        let mut loop_keys_on: HashSet<u7> = HashSet::new();

        for note_event in Self::extract_note_events(track) {
            match note_event {
                NoteEvent::On { key, vel } => {
                    if keys_on.insert(key) && keys_on.len() >= 2 + loop_keys_on.len() {
                        if let Some(&loop_key) =
                            keys_on.iter().find(|key| !loop_keys_on.contains(key))
                        {
                            loop_keys_on.insert(loop_key);
                            tokens.push(Token::LoopStart);
                        }
                    }

                    let distance = last_key_on
                        .map(|last_key| pitch_class_distance(key, last_key))
                        .unwrap_or(0);
                    match distance {
                        0 => {}
                        1 => tokens.push(Token::Decrement((vel.as_int() + 1).div_ceil(32))),
                        2 => tokens.push(Token::Increment((vel.as_int() + 1).div_ceil(32))),
                        3 => tokens.push(Token::MoveLeft((vel.as_int() + 1).div_ceil(64))),
                        4 => tokens.push(Token::MoveRight((vel.as_int() + 1).div_ceil(64))),
                        5 => tokens.push(Token::Output),
                        6 => tokens.push(Token::Input),
                        _ => panic!("Impossible note distance"),
                    };

                    last_key_on = Some(key);
                }
                NoteEvent::Off { key } => {
                    keys_on.shift_remove(&key);
                    if loop_keys_on.remove(&key) {
                        tokens.push(Token::LoopEnd);
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Extracts the notes from the given `track` and returns them as a vector
    /// of [NoteEvent]s.
    ///
    /// When they are at the same exact delta, sorts:
    /// - Adjacent note offs before note ons
    /// - Adjacent note offs/ons in ascending order
    fn extract_note_events(track: &Track) -> Vec<NoteEvent> {
        let mut notes = Vec::new();
        let mut note_offs = Vec::new();
        let mut note_ons = Vec::new();

        for event in track {
            let message = match event.kind {
                TrackEventKind::Midi { message, .. } => message,
                _ => continue,
            };

            if event.delta > 0 {
                note_offs.sort_by_key(|event| match event {
                    NoteEvent::Off { key } => *key,
                    NoteEvent::On { .. } => panic!("Impossible"),
                });
                notes.append(&mut note_offs);

                note_ons.sort_by_key(|event| match event {
                    NoteEvent::Off { .. } => panic!("Impossible"),
                    NoteEvent::On { key, .. } => *key,
                });
                notes.append(&mut note_ons);
            }

            match message {
                MidiMessage::NoteOff { key, .. } => note_offs.push(NoteEvent::Off { key }),
                MidiMessage::NoteOn { key, vel } => note_ons.push(NoteEvent::On { key, vel }),
                _ => {}
            }
        }

        notes.append(&mut note_offs);
        notes.append(&mut note_ons);

        notes
    }

    /// Returns a map from loop start and end `tokens` indices to loop end and
    /// start `tokens` indices, respectively.
    ///
    /// Panics if there are unmatched loop start or end tokens.
    fn find_loop_boundaries(tokens: &[Token]) -> HashMap<usize, usize> {
        let mut loop_boundary_indices: HashMap<usize, usize> = HashMap::new();

        let mut loop_start_indices = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            match token {
                Token::LoopStart => loop_start_indices.push(index),
                Token::LoopEnd => {
                    let start_index = loop_start_indices.pop().expect("Loop end without start");
                    loop_boundary_indices.insert(start_index, index);
                    loop_boundary_indices.insert(index, start_index);
                }
                _ => {}
            }
        }

        if !loop_start_indices.is_empty() {
            panic!("Loop start without end");
        }

        loop_boundary_indices
    }

    /// Converts the [Program] to MIDI bytes.
    pub fn to_midi(&self, bpm: u32) -> Vec<u8> {
        ProgramToMidiConverter::default().convert(self, bpm)
    }

    /// Converts the [Program] to a BF program string.
    pub fn to_bf(&self) -> String {
        let mut strings = Vec::new();
        for token in self.tokens.iter() {
            match token {
                Token::Decrement(amount) => strings.push("-".repeat((*amount).into())),
                Token::Increment(amount) => strings.push("+".repeat((*amount).into())),
                Token::MoveLeft(amount) => strings.push("<".repeat((*amount).into())),
                Token::MoveRight(amount) => strings.push(">".repeat((*amount).into())),
                Token::Output => strings.push(".".to_string()),
                Token::Input => strings.push(",".to_string()),
                Token::LoopStart => strings.push("[".to_string()),
                Token::LoopEnd => strings.push("]".to_string()),
            }
        }

        strings.join("")
    }
}

/// A struct representing the event of pressing or releasing a note.
enum NoteEvent {
    Off { key: u7 },
    On { key: u7, vel: u7 },
}

/// Calculates the pitch class distance between two keys.
fn pitch_class_distance(key1: u7, key2: u7) -> u8 {
    let distance: u8 = pitch_class(key1).abs_diff(pitch_class(key2));
    let alternate_distance = 12 - distance;
    distance.min(alternate_distance)
}

/// Returns the pitch class of a key, which is the key's value modulo 12.
fn pitch_class(key: u7) -> u8 {
    key.as_int() % 12
}

/// A token representing a single instruction in a Schoenberg program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Decrement(u8),
    Increment(u8),
    MoveLeft(u8),
    MoveRight(u8),
    Output,
    Input,
    LoopStart,
    LoopEnd,
}

/// An error that can occur when compiling MIDI bytes to a [Program].
#[derive(Debug, Clone)]
pub enum Error {
    Parse(midly::Error),
    MultipleTracks(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Parse(error) => write!(f, "{}", error),
            Error::MultipleTracks(count) => {
                write!(f, "Only one track is allowed, but got {}", count)
            }
        }
    }
}

impl error::Error for Error {}

/// A struct for holding the current state of a [Program] to MIDI conversion.
struct ProgramToMidiConverter {
    timestamp: u28,
    midi_note_ranges: HashMap<u7, Vec<NoteRange>>,
    last_note_range: NoteRange,
    last_direction: Direction,
    loop_keys: IndexSet<u7>,
}

/// A struct representing a key that was pressed and released.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NoteRange {
    key: u7,
    vel: u7,
    start: u28,
    end: u28,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Down,
    Up,
}

impl Default for ProgramToMidiConverter {
    fn default() -> Self {
        Self {
            timestamp: 0.into(),
            midi_note_ranges: HashMap::new(),
            last_note_range: NoteRange {
                key: 0.into(),
                vel: 0.into(),
                start: 0.into(),
                end: 0.into(),
            },
            last_direction: Direction::Down,
            loop_keys: IndexSet::new(),
        }
    }
}

impl ProgramToMidiConverter {
    const TICKS_PER_QUARTER: u16 = 480;

    const SIXTEENTH_DELTA: u28 = u28::new((Self::TICKS_PER_QUARTER / 4) as u32);
    const EIGHTH_DELTA: u28 = u28::new((Self::TICKS_PER_QUARTER / 2) as u32);
    const QUARTER_DELTA: u28 = u28::new(Self::TICKS_PER_QUARTER as u32);
    const HALF_DELTA: u28 = u28::new((Self::TICKS_PER_QUARTER * 2) as u32);

    const DEFAULT_VEL: u7 = u7::new(63);

    fn convert(&mut self, program: &Program, bpm: u32) -> Vec<u8> {
        self.play_notes(program);

        let header = Header {
            format: Format::SingleTrack,
            timing: Timing::Metrical(Self::TICKS_PER_QUARTER.into()),
        };
        let bf = program.to_bf();
        let smf = Smf {
            header,
            tracks: vec![self.build_track(&bf, bpm)],
        };

        let mut midi_data = Vec::new();
        smf.write(&mut midi_data).unwrap();
        midi_data
    }

    /// Plays notes corresponding to the tokens of the given `program`.
    fn play_notes(&mut self, program: &Program) {
        // Start on Middle C
        self.note_on(60.into(), Self::DEFAULT_VEL);

        for token in Self::split_program_tokens(program).iter() {
            match token {
                Token::Decrement(amount) => {
                    let next_key = self.next_key(1.into());
                    self.note_on(next_key, (amount * 32 - 1).into());
                }
                Token::Increment(amount) => {
                    let next_key = self.next_key(2.into());
                    self.note_on(next_key, (amount * 32 - 1).into());
                }
                Token::MoveLeft(amount) => {
                    let next_key = self.next_key(3.into());
                    self.note_on(next_key, (amount * 64 - 1).into());
                }
                Token::MoveRight(amount) => {
                    let next_key = self.next_key(4.into());
                    self.note_on(next_key, (amount * 64 - 1).into());
                }
                Token::Output => {
                    let next_key = self.next_key(5.into());
                    self.note_on(next_key, Self::DEFAULT_VEL);
                }
                Token::Input => {
                    let next_key = self.next_key(6.into());
                    self.note_on(next_key, Self::DEFAULT_VEL);
                }
                Token::LoopStart => {
                    let last_key = self.last_note_range.key;
                    if self.loop_keys.insert(last_key) {
                        // The last key isn't already a loop key, so we can use
                        // it as the loop key.
                        continue;
                    }

                    // Otherwise, we need to find a new key with the same pitch
                    // class distance for the purposes of looping.
                    let loop_key = [Direction::Up, Direction::Down]
                        .iter()
                        .cycle()
                        .enumerate()
                        .map(|(index, &direction)| {
                            let delta = 12 * (index + 1);
                            match direction {
                                Direction::Down => last_key - (delta as u8).into(),
                                Direction::Up => last_key + (delta as u8).into(),
                            }
                        })
                        .find(|&loop_key| self.loop_keys.insert(loop_key))
                        .unwrap();
                    self.note_on(loop_key, Self::DEFAULT_VEL);
                }
                Token::LoopEnd => {
                    self.loop_keys.pop();
                }
            }

            for loop_key in self.loop_keys.clone() {
                self.note_continue(loop_key);
            }
        }
    }

    /// Splits the given `program` tokens into smaller increments so that each
    /// token an be represented by a single MIDI note, and isn't too loud.
    fn split_program_tokens(program: &Program) -> Vec<Token> {
        program
            .tokens
            .iter()
            .flat_map(|&token| match token {
                Token::Decrement(amount)
                | Token::Increment(amount)
                | Token::MoveLeft(amount)
                | Token::MoveRight(amount) => {
                    let mut tokens = Vec::new();
                    let mut current_amount = amount;
                    while current_amount > 0 {
                        let amount = 2.min(current_amount);
                        current_amount -= amount;
                        tokens.push(match token {
                            Token::Decrement(_) => Token::Decrement(amount),
                            Token::Increment(_) => Token::Increment(amount),
                            Token::MoveLeft(_) => Token::MoveLeft(amount),
                            Token::MoveRight(_) => Token::MoveRight(amount),
                            _ => unreachable!(),
                        });
                    }
                    tokens
                }
                _ => vec![token],
            })
            .collect()
    }

    /// Presses and releases the given `key` with the given `vel`.
    fn note_on(&mut self, key: u7, vel: u7) {
        let start = self.timestamp;
        self.timestamp += Self::next_key_delta();
        let end = self.timestamp;
        self.timestamp += Self::next_rest_delta();

        self.last_note_range = NoteRange {
            key,
            vel,
            start,
            end,
        };
        self.midi_note_ranges
            .entry(key)
            .or_default()
            .push(self.last_note_range.clone());
    }

    /// Returns the delta to use for the next key press.
    fn next_key_delta() -> u28 {
        let mut rng = rand::rng();
        let weighted_deltas = [
            (Self::SIXTEENTH_DELTA, 12),
            (Self::EIGHTH_DELTA, 11),
            (Self::QUARTER_DELTA, 10),
            (Self::HALF_DELTA, 1),
        ];

        let &(delta, _) = weighted_deltas
            .choose_weighted(&mut rng, |&(_, weight)| weight)
            .unwrap();
        delta
    }

    /// Returns the delta to use for the next rest.
    fn next_rest_delta() -> u28 {
        let mut rng = rand::rng();
        let weighted_deltas = [
            (Self::SIXTEENTH_DELTA, 12),
            (Self::EIGHTH_DELTA, 11),
            (Self::QUARTER_DELTA, 10),
            // Allow legato.
            (0.into(), 4),
            (Self::HALF_DELTA, 1),
        ];

        let &(delta, _) = weighted_deltas
            .choose_weighted(&mut rng, |&(_, weight)| weight)
            .unwrap();
        delta
    }

    /// Modifies the most recently playing of the given `key` to end at the
    /// current timestamp.
    ///
    /// Panics if the given `key` was never played.
    fn note_continue(&mut self, key: u7) {
        self.midi_note_ranges
            .get_mut(&key)
            .and_then(|midi_notes| midi_notes.last_mut())
            .unwrap()
            .end = self.last_note_range.end;
    }

    /// Returns a next key to play that has the given `target_distance` for the
    /// pitch class distance from the last played key.
    fn next_key(&mut self, target_distance: u7) -> u7 {
        let last_key = self.last_note_range.key;
        let direction = if last_key < 50 {
            // Don't go too low.
            Direction::Up
        } else if last_key > 85 {
            // Don't go too high.
            Direction::Down
        } else {
            let mut rng = rand::rng();
            // Slight preference for continuing in the same direction as before.
            match self.last_direction {
                Direction::Down => [(Direction::Down, 3), (Direction::Up, 2)],
                Direction::Up => [(Direction::Down, 2), (Direction::Up, 3)],
            }
            .choose_weighted(&mut rng, |&(_, weight)| weight)
            .unwrap()
            .0
        };
        self.last_direction = direction;

        let next_key_downs =
            (1..=11).map(|distance| last_key.as_int().saturating_sub(distance).into());
        let next_key_ups = (1..=11).map(|distance| last_key + distance.into());
        let next_keys: Vec<_> = match direction {
            Direction::Down => next_key_downs.chain(next_key_ups).collect(),
            Direction::Up => next_key_ups.chain(next_key_downs).collect(),
        };

        next_keys
            .iter()
            .filter(|&next_key| pitch_class_distance(last_key, *next_key) == target_distance)
            .cycle()
            .enumerate()
            .map(|(index, next_key)| {
                let cycle_index = (index / next_keys.len()) as u8;
                let delta = cycle_index * 12;
                match direction {
                    Direction::Down => {
                        if cycle_index % 2 == 0 {
                            next_key.as_int() + delta
                        } else {
                            next_key.as_int() - delta
                        }
                    }
                    Direction::Up => {
                        if cycle_index % 2 == 0 {
                            next_key.as_int() - delta
                        } else {
                            next_key.as_int() + delta
                        }
                    }
                }
                .into()
            })
            .find(|next_key| !self.loop_keys.contains(next_key))
            .unwrap()
    }

    /// Builds a MIDI [Track] from the played notes.
    fn build_track<'a>(&self, bf: &'a str, bpm: u32) -> Track<'a> {
        let mut track = vec![
            TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::ProgramName(b"Schoenberg")),
            },
            TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::TrackName(bf.as_bytes())),
            },
            TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::Tempo((60_000_000 / bpm).into())),
            },
            TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
            },
        ];

        let mut midi_messages: Vec<_> = self
            .midi_note_ranges
            .values()
            .flatten()
            .flat_map(|midi_note| {
                vec![
                    TrackEvent {
                        delta: midi_note.start,
                        kind: TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::NoteOn {
                                key: midi_note.key,
                                vel: midi_note.vel,
                            },
                        },
                    },
                    TrackEvent {
                        delta: midi_note.end,
                        kind: TrackEventKind::Midi {
                            channel: 0.into(),
                            message: MidiMessage::NoteOff {
                                key: midi_note.key,
                                vel: midi_note.vel,
                            },
                        },
                    },
                ]
            })
            .collect();
        midi_messages.sort_by_key(|event| event.delta);

        // The deltas are absolute, but MIDI requires them to be relative to the
        // previous note, so we make them relative here.
        let mut timestamp = 0.into();
        for event in &mut midi_messages {
            let delta = event.delta;
            event.delta -= timestamp;
            timestamp = delta;
        }

        track.extend(midi_messages);
        track.push(TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });

        track
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;

    #[test]
    fn test_sample_roundtrip() {
        let bf_programs: Vec<String> = fs::read_dir("./samples")
            .unwrap()
            .map(|res| res.unwrap())
            .filter(|item| {
                item.file_type().unwrap().is_file()
                    && item.file_name().into_string().unwrap().ends_with(".b")
            })
            .map(|item| fs::read_to_string(item.path()).unwrap())
            .map(|bf| bf.lines().filter(|line| !line.starts_with("#")).collect())
            .collect();
        println!("{} BF programs", bf_programs.len());

        for (index, bf) in bf_programs.into_iter().enumerate() {
            println!("{}: {}", index, bf);

            let roundtripped_bf = Program::from_midi(&Program::from_bf(&bf).to_midi(120))
                .unwrap()
                .to_bf();

            assert_eq!(roundtripped_bf, bf);
        }
    }

    fn bf_strategy() -> impl Strategy<Value = String> {
        let char = prop_oneof![
            Just("+".to_string()),
            Just("-".to_string()),
            Just(">".to_string()),
            Just("<".to_string()),
            Just(".".to_string()),
            Just(",".to_string()),
        ];
        char.prop_recursive(8, 30, 5, |inner| {
            prop::collection::vec(inner.clone(), 1..=10)
                .prop_map(|chars| ["[", &chars.join(""), "]"].join(""))
        })
    }

    proptest! {
        #[test]
        fn test_bf_roundtrip_property(bf in bf_strategy()) {
            let roundtripped_bf = Program::from_midi(&Program::from_bf(&bf).to_midi(120))
                .unwrap()
                .to_bf();

            prop_assert_eq!(roundtripped_bf, bf);
        }
    }
}
