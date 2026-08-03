use crate::{Result, RuntimeError};

pub const EOS_TOKEN: u32 = 0;
pub const BOS_TOKEN: u32 = 1;

#[derive(Debug, Clone, Copy, Default)]
pub struct TinyTokenizer;

impl TinyTokenizer {
    pub fn encode(self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(text.chars().count() + 1);
        tokens.push(BOS_TOKEN);
        for (byte_offset, character) in text.char_indices() {
            let token = match character {
                'a'..='z' => u32::from(character) - u32::from('a') + 2,
                ' ' => 28,
                '.' => 29,
                ',' => 30,
                '?' => 31,
                _ => return Err(RuntimeError::UnsupportedCharacter { byte_offset }),
            };
            tokens.push(token);
        }
        Ok(tokens)
    }

    pub fn decode(self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        for token in tokens {
            match *token {
                EOS_TOKEN => break,
                BOS_TOKEN => {}
                2..=27 => {
                    let offset = u8::try_from(*token - 2).expect("token range is bounded");
                    text.push(char::from(b'a' + offset));
                }
                28 => text.push(' '),
                29 => text.push('.'),
                30 => text.push(','),
                31 => text.push('?'),
                token => {
                    return Err(RuntimeError::InvalidToken {
                        token,
                        vocab_size: 32,
                    });
                }
            }
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_supported_text() {
        let tokenizer = TinyTokenizer;
        let encoded = tokenizer.encode("moe, yes?").unwrap();
        assert_eq!(encoded[0], BOS_TOKEN);
        assert_eq!(tokenizer.decode(&encoded).unwrap(), "moe, yes?");
        assert_eq!(tokenizer.encode("").unwrap(), [BOS_TOKEN]);
        let all_text = "abcdefghijklmnopqrstuvwxyz .,?";
        assert_eq!(
            tokenizer.encode(all_text).unwrap(),
            (1..32).collect::<Vec<_>>()
        );
        assert_eq!(
            tokenizer.decode(&(1..32).collect::<Vec<_>>()).unwrap(),
            all_text
        );
    }

    #[test]
    fn unsupported_character_reports_utf8_byte_offset() {
        let error = TinyTokenizer.encode("aé").unwrap_err();
        assert_eq!(error, RuntimeError::UnsupportedCharacter { byte_offset: 1 });
    }

    #[test]
    fn eos_stops_decode() {
        assert_eq!(TinyTokenizer.decode(&[2, EOS_TOKEN, 3]).unwrap(), "a");
        assert!(matches!(
            TinyTokenizer.decode(&[32]),
            Err(RuntimeError::InvalidToken { .. })
        ));
    }
}
