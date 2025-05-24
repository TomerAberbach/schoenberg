use clap::Parser;
use clap_stdin::{FileOrStdin, MaybeStdin};
use schoenberg::Program;
use std::{
    error,
    io::{self, Read, Write},
    process,
    str::FromStr,
};

mod schoenberg;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    subcommand: Subcommand,
}

#[derive(Debug, clap::Subcommand, Clone)]
enum Subcommand {
    Run {
        #[arg(default_value = "-")]
        midi: FileOrStdin,

        #[arg(long = "input", short = 'i', default_value_t = MaybeStdin::from_str("").unwrap())]
        input: MaybeStdin<String>,
    },
    FromBf {
        #[arg(default_value = "-")]
        input: MaybeStdin<String>,
    },
    ToBf {
        #[arg(default_value = "-")]
        midi: FileOrStdin,
    },
}

fn main() {
    let args = Args::parse();
    let res = match args.subcommand {
        Subcommand::Run { midi, input } => run(midi, &input),
        Subcommand::FromBf { input } => from_bf(&input),
        Subcommand::ToBf { midi } => to_bf(midi),
    };
    if let Err(err) = res {
        println!("{}", err);
        process::exit(1);
    }
}

fn run(midi: FileOrStdin, input: &str) -> Result<(), Box<dyn error::Error>> {
    let midi_bytes = midi
        .into_reader()?
        .bytes()
        .collect::<Result<Vec<u8>, _>>()?;
    let program = Program::from_midi(&midi_bytes)?;
    let output = program.run(input);
    println!("{}", output);

    Ok(())
}

fn from_bf(input: &str) -> Result<(), Box<dyn error::Error>> {
    let midi_bytes = Program::from_bf(input).to_midi();

    let mut stdout = io::stdout();
    stdout.write_all(&midi_bytes)?;
    stdout.flush()?;

    Ok(())
}

fn to_bf(midi: FileOrStdin) -> Result<(), Box<dyn error::Error>> {
    let midi_bytes = midi
        .into_reader()?
        .bytes()
        .collect::<Result<Vec<u8>, _>>()?;
    let program = Program::from_midi(&midi_bytes)?;
    println!("{}", program.to_bf());

    Ok(())
}
