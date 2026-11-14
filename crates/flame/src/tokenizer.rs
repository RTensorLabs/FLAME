//! Tokenizers.
//!
//! * [`Tokenizer::Byte`] — every UTF-8 byte is a token (vocab 256 + specials). Zero training,
//!   perfect for overfitting tests and character-level synthetic tasks.
//! * [`Tokenizer::Bpe`]  — byte-level Byte-Pair Encoding trained from your own text
//!   (the same family of tokenizer used by GPT-2 / LLaMA-3). Roughly 3–4× fewer tokens than
//!   bytes on English or Rust code, which directly means 3–4× less training compute per byte.
//!
//! Token id layout (both kinds):
//!   0..=255      raw bytes
//!   256          `<|endoftext|>` (document separator / end of sequence)
//!   257          `<|pad|>`
//!   258..        BPE merges, in the order they were learned

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;

pub const EOS_ID: u32 = 256;
pub const PAD_ID: u32 = 257;
pub const FIRST_MERGE_ID: u32 = 258;
pub const EOS_TEXT: &str = "<|endoftext|>";
pub const PAD_TEXT: &str = "<|pad|>";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Tokenizer {
    Byte,
    Bpe(BpeTokenizer),
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading tokenizer {}", path.display()))?;
        Self::from_json(&text)
    }

    /// Parses a tokenizer from the JSON text `save` writes (used where there is no file
    /// system, e.g. the browser demo).
    pub fn from_json(text: &str) -> Result<Self> {
        let mut tok: Tokenizer = serde_json::from_str(text)?;
        if let Tokenizer::Bpe(b) = &mut tok {
            b.rebuild_tables();
        }
        Ok(tok)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Byte => FIRST_MERGE_ID as usize,
            Tokenizer::Bpe(b) => FIRST_MERGE_ID as usize + b.merges.len(),
        }
    }

    pub fn eos_id(&self) -> u32 {
        EOS_ID
    }

    pub fn pad_id(&self) -> u32 {
        PAD_ID
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Tokenizer::Byte => text.bytes().map(|b| b as u32).collect(),
            Tokenizer::Bpe(b) => b.encode(text),
        }
    }

    /// Decodes ids to text. Invalid UTF-8 (e.g. a sample cut in the middle of a multi-byte
    /// character) is replaced by U+FFFD instead of failing.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            self.push_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn push_bytes(&self, id: u32, out: &mut Vec<u8>) {
        match id {
            0..=255 => out.push(id as u8),
            EOS_ID => out.extend_from_slice(EOS_TEXT.as_bytes()),
            PAD_ID => {}
            _ => match self {
                Tokenizer::Byte => {}
                Tokenizer::Bpe(b) => {
                    if let Some(bytes) = b.token_bytes.get((id - FIRST_MERGE_ID) as usize) {
                        out.extend_from_slice(bytes)
                    }
                }
            },
        }
    }
}

/// Splits text into "words" before BPE, so merges never cross word boundaries.
///
/// Rules (simplified GPT-2 style, tuned for prose *and* source code):
/// * an identifier-like run `[A-Za-z_][A-Za-z0-9_]*` (or any non-ASCII alphabetic run),
/// * a run of digits (at most 3 per chunk, so numbers don't explode the vocab),
/// * a run of punctuation,
///
/// each optionally preceded by ONE space; newlines + following indentation form one chunk.
pub fn pretokenize(text: &str) -> Vec<&str> {
    #[derive(PartialEq, Clone, Copy)]
    enum Class {
        Word,
        Digit,
        Punct,
        Space,
        Newline,
    }
    fn class(c: char) -> Class {
        if c == '\n' || c == '\r' {
            Class::Newline
        } else if c.is_whitespace() {
            Class::Space
        } else if c.is_ascii_digit() {
            Class::Digit
        } else if c.is_alphanumeric() || c == '_' {
            Class::Word
        } else {
            Class::Punct
        }
    }

    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut i = 0;
    while i < n {
        let start = i;
        let mut c = class(chars[i].1);
        // A single ordinary space glues onto the following non-space token (" fn", " =").
        if chars[i].1 == ' ' && i + 1 < n && !matches!(class(chars[i + 1].1), Class::Space | Class::Newline) {
            i += 1;
            c = class(chars[i].1);
        }
        match c {
            Class::Word => {
                // identifiers may contain digits after the first character
                while i < n && matches!(class(chars[i].1), Class::Word | Class::Digit) {
                    i += 1;
                }
            }
            Class::Digit => {
                let s = i;
                while i < n && class(chars[i].1) == Class::Digit && i - s < 3 {
                    i += 1;
                }
            }
            Class::Punct => {
                while i < n && class(chars[i].1) == Class::Punct {
                    i += 1;
                }
            }
            Class::Newline => {
                // "\n" plus the indentation of the next line
                while i < n && class(chars[i].1) == Class::Newline {
                    i += 1;
                }
                while i < n && (chars[i].1 == ' ' || chars[i].1 == '\t') {
                    i += 1;
                }
            }
            Class::Space => {
                // leave the last space to be glued onto the next word
                while i < n && class(chars[i].1) == Class::Space {
                    if chars[i].1 == ' '
                        && i + 1 < n
                        && !matches!(class(chars[i + 1].1), Class::Space | Class::Newline)
                        && i > start
                    {
                        break;
                    }
                    i += 1;
                }
            }
        }
        if i == start {
            i += 1;
        }
        out.push(&text[byte_at(start)..byte_at(i)]);
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Merge rules in learned order; merge `i` creates token id `FIRST_MERGE_ID + i`.
    pub merges: Vec<(u32, u32)>,
    #[serde(skip)]
    ranks: HashMap<(u32, u32), u32>,
    #[serde(skip)]
    token_bytes: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut t = BpeTokenizer { merges, ranks: HashMap::new(), token_bytes: Vec::new() };
        t.rebuild_tables();
        t
    }

    fn rebuild_tables(&mut self) {
        self.ranks = self.merges.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        self.token_bytes.clear();
        for i in 0..self.merges.len() {
            let (a, b) = self.merges[i];
            let mut bytes = self.bytes_of(a);
            bytes.extend(self.bytes_of(b));
            self.token_bytes.push(bytes);
        }
    }

    fn bytes_of(&self, id: u32) -> Vec<u8> {
        match id {
            0..=255 => vec![id as u8],
            EOS_ID | PAD_ID => vec![],
            _ => self.token_bytes[(id - FIRST_MERGE_ID) as usize].clone(),
        }
    }

    /// Trains BPE on `text` until the vocabulary has `vocab_size` entries.
    ///
    /// Uses the standard incremental algorithm: pair counts are updated only for the words
    /// that contain the merged pair, and the best pair is found with a lazy max-heap.
    pub fn train(text: &str, vocab_size: usize, verbose: bool) -> Result<Self> {
        anyhow::ensure!(vocab_size > FIRST_MERGE_ID as usize, "vocab_size must be > {FIRST_MERGE_ID}");
        anyhow::ensure!(vocab_size <= u16::MAX as usize + 1, "vocab_size must fit in u16 (≤ 65536)");
        let n_merges = vocab_size - FIRST_MERGE_ID as usize;

        // 1. Count unique pre-tokenized words.
        let mut counts: HashMap<&str, i64> = HashMap::new();
        for w in pretokenize(text) {
            *counts.entry(w).or_default() += 1;
        }
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(counts.len());
        let mut freqs: Vec<i64> = Vec::with_capacity(counts.len());
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort(); // deterministic training
        for (w, c) in sorted {
            words.push(w.bytes().map(|b| b as u32).collect());
            freqs.push(c);
        }

        // 2. Initial pair statistics.
        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for (wi, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                let pair = (p[0], p[1]);
                *pair_counts.entry(pair).or_default() += freqs[wi];
                where_.entry(pair).or_default().insert(wi);
            }
        }

        #[derive(PartialEq, Eq)]
        struct Item(i64, (u32, u32));
        impl Ord for Item {
            fn cmp(&self, o: &Self) -> Ordering {
                // highest count first; ties broken by the smallest pair (deterministic)
                self.0.cmp(&o.0).then_with(|| o.1.cmp(&self.1))
            }
        }
        impl PartialOrd for Item {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        let mut heap: BinaryHeap<Item> = pair_counts.iter().map(|(&p, &c)| Item(c, p)).collect();

        let mut merges = Vec::with_capacity(n_merges);
        while merges.len() < n_merges {
            let Some(Item(count, pair)) = heap.pop() else { break };
            let current = *pair_counts.get(&pair).unwrap_or(&0);
            if count != current {
                // stale heap entry — re-insert with the fresh count
                if current > 0 {
                    heap.push(Item(current, pair));
                }
                continue;
            }
            if count < 2 {
                break; // nothing worth merging any more
            }
            let new_id = FIRST_MERGE_ID + merges.len() as u32;
            merges.push(pair);
            let affected: Vec<usize> = where_.remove(&pair).map(|s| s.into_iter().collect()).unwrap_or_default();
            let mut touched: HashSet<(u32, u32)> = HashSet::new();
            for wi in affected {
                let w = &words[wi];
                let f = freqs[wi];
                // remove old pairs of this word
                for p in w.windows(2) {
                    let pp = (p[0], p[1]);
                    *pair_counts.get_mut(&pp).unwrap() -= f;
                    touched.insert(pp);
                }
                // merge
                let mut nw = Vec::with_capacity(w.len());
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && (w[i], w[i + 1]) == pair {
                        nw.push(new_id);
                        i += 2;
                    } else {
                        nw.push(w[i]);
                        i += 1;
                    }
                }
                // add new pairs
                for p in nw.windows(2) {
                    let pp = (p[0], p[1]);
                    *pair_counts.entry(pp).or_default() += f;
                    where_.entry(pp).or_default().insert(wi);
                    touched.insert(pp);
                }
                words[wi] = nw;
            }
            pair_counts.remove(&pair);
            for p in touched {
                if let Some(&c) = pair_counts.get(&p) {
                    if c > 0 && p != pair {
                        heap.push(Item(c, p));
                    }
                }
            }
            if verbose && merges.len() % 1000 == 0 {
                eprintln!("  learned {} / {} merges", merges.len(), n_merges);
            }
        }
        if merges.len() < n_merges && verbose {
            eprintln!("  stopped early at {} merges (not enough repeated pairs in the sample)", merges.len());
        }
        Ok(Self::from_merges(merges))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        for w in pretokenize(text) {
            if let Some(ids) = cache.get(w) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.encode_word(w.as_bytes());
            out.extend_from_slice(&ids);
            cache.insert(w, ids);
        }
        out
    }

    fn encode_word(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
        // Repeatedly apply the earliest-learned merge present in the word.
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((rank, _)) = best else { break };
            let pair = self.merges[rank as usize];
            let new_id = FIRST_MERGE_ID + rank;
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_is_lossless() {
        let text = "fn main() {\n    let mut x = 42_000;\n    println!(\"{x}\");  // done\n}\n";
        assert_eq!(pretokenize(text).concat(), text);
        let prose = "Once upon a time,  there was   a café. 12345!";
        assert_eq!(pretokenize(prose).concat(), prose);
    }

    #[test]
    fn bpe_roundtrip_and_compresses() {
        let text = "the cat sat on the mat. the cat ate the rat. ".repeat(50)
            + "fn main() { let x = 1; let y = x + 1; println!(\"{}\", y); }\n";
        let bpe = BpeTokenizer::train(&text, 300, false).unwrap();
        let tok = Tokenizer::Bpe(bpe);
        let ids = tok.encode(&text);
        assert_eq!(tok.decode(&ids), text);
        assert!(ids.len() * 2 < text.len(), "BPE should compress repetitive text");
        // unseen text (incl. non-ASCII) must still round-trip through byte fallback
        let unseen = "Grüße, ζ! 🚀 impl Trait for T {}";
        assert_eq!(tok.decode(&tok.encode(unseen)), unseen);
    }

    #[test]
    fn save_load_roundtrip() {
        let bpe = BpeTokenizer::train(&"hello world ".repeat(100), 270, false).unwrap();
        let tok = Tokenizer::Bpe(bpe);
        let dir = std::env::temp_dir().join(format!("flame_tok_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tok.json");
        tok.save(&p).unwrap();
        let loaded = Tokenizer::load(&p).unwrap();
        assert_eq!(loaded.encode("hello world hello"), tok.encode("hello world hello"));
        assert_eq!(loaded.vocab_size(), tok.vocab_size());
    }
}
