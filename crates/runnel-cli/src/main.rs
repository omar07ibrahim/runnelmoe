use std::{error::Error, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand, ValueEnum};
use runnel_fixture::{FixtureArtifact, FixtureIdentity};
use runnel_format::{Artifact, Limits};
use runnel_runtime::{EOS_TOKEN, TinyModel, TinyTokenizer};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "runnel", version, about = "RunnelMoE reference runtime")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Materialize the deterministic tiny RMOA fixture in a new directory.
    Fixture {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Generate tokens with a verified RMOA artifact.
    Generate {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
        #[arg(long, value_enum, default_value_t = Strategy::Greedy)]
        strategy: Strategy,
        #[arg(long)]
        json: bool,
    },
    /// Generate and verify the tiny artifact in a temporary directory, then run it.
    Demo {
        #[arg(long, default_value = "moe")]
        prompt: String,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Strategy {
    Greedy,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("greedy")
    }
}

#[derive(Debug, Serialize)]
struct AdapterOutput<'a> {
    id: &'a str,
    version: u64,
}

#[derive(Debug, Serialize)]
struct GenerationOutput<'a> {
    schema_version: u64,
    artifact_id: String,
    adapter: AdapterOutput<'a>,
    input_token_count: usize,
    generated_ids: Vec<u32>,
    text: String,
    stop_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct FixtureOutput {
    schema_version: u64,
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

fn main() -> ExitCode {
    match run(Arguments::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    match arguments.command {
        Command::Fixture { output, json } => {
            let fixture = FixtureArtifact::build();
            let identity = fixture.write_new(output)?;
            print_fixture(identity, json)?;
        }
        Command::Generate {
            artifact,
            prompt,
            max_new_tokens,
            strategy: Strategy::Greedy,
            json,
        } => {
            let artifact = Artifact::open(artifact, Limits::default())?;
            print_generation(&artifact, &prompt, max_new_tokens, json)?;
        }
        Command::Demo {
            prompt,
            max_new_tokens,
            json,
        } => {
            let temporary = tempfile::tempdir()?;
            let root = temporary.path().join("tiny-rmoa");
            FixtureArtifact::build().write_new(&root)?;
            let artifact = Artifact::open(root, Limits::default())?;
            print_generation(&artifact, &prompt, max_new_tokens, json)?;
        }
    }
    Ok(())
}

fn print_fixture(identity: FixtureIdentity, json: bool) -> Result<(), Box<dyn Error>> {
    let output = FixtureOutput {
        schema_version: 1,
        artifact_id: identity.artifact_id.to_string(),
        object_digest: identity.object_digest.to_string(),
        object_length: identity.object_length,
        page_table_digest: identity.page_table_digest.to_string(),
        page_table_length: identity.page_table_length,
    };
    if json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("artifact: {}", output.artifact_id);
        println!(
            "object: {} ({} bytes)",
            output.object_digest, output.object_length
        );
        println!(
            "page table: {} ({} bytes)",
            output.page_table_digest, output.page_table_length
        );
    }
    Ok(())
}

fn print_generation(
    artifact: &Artifact,
    prompt: &str,
    max_new_tokens: usize,
    json: bool,
) -> Result<(), Box<dyn Error>> {
    let tokenizer = TinyTokenizer;
    let input_ids = tokenizer.encode(prompt)?;
    let model = TinyModel::from_artifact(artifact)?;
    let generation = model.generate_greedy(&input_ids, max_new_tokens)?;
    let stop_reason = if generation.generated_tokens.last() == Some(&EOS_TOKEN) {
        "eos"
    } else {
        "max_new_tokens"
    };
    let output = GenerationOutput {
        schema_version: 1,
        artifact_id: artifact.artifact_id().to_string(),
        adapter: AdapterOutput {
            id: &artifact.manifest().adapter.id,
            version: artifact.manifest().adapter.version,
        },
        input_token_count: input_ids.len(),
        text: tokenizer.decode(&generation.generated_tokens)?,
        generated_ids: generation.generated_tokens,
        stop_reason,
    };

    if json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("artifact: {}", output.artifact_id);
        println!("generated IDs: {:?}", output.generated_ids);
        println!("text: {}", output.text);
        println!("stop reason: {}", output.stop_reason);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_contract_is_stable() {
        let fixture = FixtureArtifact::build();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        let model = TinyModel::from_artifact(&artifact).unwrap();
        let input = TinyTokenizer.encode("moe").unwrap();
        let generated = model.generate_greedy(&input, 4).unwrap();
        assert_eq!(generated.generated_tokens, [15, 11, 20, 9]);
        assert_eq!(
            TinyTokenizer.decode(&generated.generated_tokens).unwrap(),
            "njsh"
        );
    }
}
