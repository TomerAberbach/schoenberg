<h1 align="center">
  Schoenberg
</h1>

<div align="center">
  <a href="https://github.com/TomerAberbach/schoenberg/actions">
    <img src="https://github.com/TomerAberbach/schoenberg/workflows/CI/badge.svg" alt="CI" />
  </a>
  <a href="https://github.com/sponsors/TomerAberbach">
    <img src="https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86" alt="Sponsor" />
  </a>
</div>

<div align="center">
  The MIDI Esoteric Programming Language
</div>

## Huh?

Schoenberg is an
[esoteric programming language](https://en.wikipedia.org/wiki/Esoteric_programming_language)
where programs are written as [MIDI](https://en.wikipedia.org/wiki/MIDI) files.

This repository contains:
- The programming language's interpreter
- Utilities for converting between Schoenberg and
  [Brainfuck](https://en.wikipedia.org/wiki/Brainfuck) programs
- [Sample programs and their generated audio files](./samples)

To learn more about the language itself
[check out my website](https://tomeraberba.ch/schoenberg)!

## Install

```sh
$ cargo install schoenberg
```

Make sure `~/.cargo/bin` is in your `PATH`.

## Usage

```sh
$ schoenberg --help
```

```
Usage: schoenberg <COMMAND>

Commands:
  run      Run a Schoenberg MIDI program
  from-bf  Convert a BF program to a Schoenberg MIDI program
  to-bf    Convert a Schoenberg MIDI program to a BF program
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Contributing

Stars are always welcome!

For bugs and feature requests,
[please create an issue](https://github.com/TomerAberbach/schoenberg/issues/new).

## License

[MIT](https://github.com/TomerAberbach/schoenberg/blob/main/license)
© [Tomer Aberbach](https://github.com/TomerAberbach)
