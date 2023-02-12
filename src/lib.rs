#![feature(iter_intersperse)]
#![feature(hash_raw_entry)]
//! rdrain implements the [Drain]() algorithm for automatic log parsing.
//! # Example:
//! 

//! # Vocabulary
//! A **log record** is an entry in a text file, typically one-line but doesn't have to be.
//!
//!   E.g. `[Thu Jun 09 06:07:05 2005] [notice] Digest logline here: done`
//!
//!
//! A **log template** is the string template that was used to format log record.
//!
//! For example in Python format, that would look like this:
//!
//!   `"[{date}] [{log_level}] Digest logline here: {status}".format(...)`
//!
//! Or in the syntax output by drain.py and drain3.py:
//!
//!   `"[<date>] [<log_level>] Digest logline here: <*>"`
//!
//! # TODO
//! The first drain allowed `split_line_provided`, which let you write a simple token-mapper like this:
//!
//!   `<timestamp> <loglevel> <content>`
//!
//! And then drain would only apply its logic to `<content>`.
//!
//! Drain3 appears to have dropped this in favor of preprocessing on the user-code side, which is fair enough.
//!
//! Drain3 allows "masking", which appears to be for recognizing values like IPs or numbers.
//! We have preliminary support for masking but it's not configurable from outside of the class and the user interface
//! to it is not yet defined.

use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::fmt;
use std::error::Error;
use thiserror::Error;
use log::{debug, info, error};
use rustc_hash::FxHashMap;
use json_in_type::*;
use json_in_type::list::ToJSONList;
use std::iter::zip;

/// In the process of parsing, the drain algo populates a ParseTree. This tree could be saved
/// and re-used on the next run, to avoid "forgetting" the previously recognized log templates.
#[derive(Default)]
pub struct ParseTree {
    root: TreeRoot,
    next_cluster_id: usize,
    // = TreeRoot::new();
}

fn zip_tokens_and_template<'c>(
    templatetokens: &[LogTemplateItem],
    logtokens: &[TokenParse<'c>],
    results: &mut Vec<&'c str>
) {
    results.clear();
    for (template_token, log_token) in zip(templatetokens, logtokens) {
        match template_token {
            LogTemplateItem::StaticToken(_) => {}
            LogTemplateItem::Value => match log_token {
                TokenParse::Token(v) => results.push(*v),
                TokenParse::MaskedValue(v) => results.push(*v),
            },
        }
    }
}

/// The elements in a LogTemplate (not a record).
/// Given a log-template (in string form) like this,
///
///   `"[{date}] [{log_level}] Digest logline here: {status}"`
///
///  the parsed rich-type form would be:
///
///   `[Value, Value, StaticToken("Digest"), StaticToken("logline"), StaticToken("here:"), Value]`
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum LogTemplateItem {
    StaticToken(String), // Owned because we need to store it.
    Value,               // Python port used "<*>" instead.
}

impl fmt::Display for LogTemplateItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::StaticToken(s) => s,
                Self::Value => "<*>",
            }
        )
    }
}

#[derive(Debug)]
enum TokenParse<'a> {
    Token(&'a str),
    MaskedValue(&'a str),
}
#[derive(Debug)]
enum Preprocessed<'a> {
    Segment(&'a str),
    Value(&'a str),
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("couldn't parse line with user defined template, multiline log msg?")]
    NoTokensInRecord,
}

/// For each log record, contains template_id the record belongs to, and `values` used to create the record.
#[derive(Debug)]
pub struct RecordParsed<'a> {
    /// Maps 1:1 to the order of NewTemplates recognized.
    /// Use this to map to any state you stored when recvd NewTemplate.
    pub template_id: usize,
    /// The values used to populate the record from the template. Given the following template:
    ///
    ///   `[Value, Value, StaticToken("Digest"), StaticToken("logline"), StaticToken("here:"), Value]`
    /// values for a particular record might be:
    ///
    /// E.g. ["Thu Jun 09 06:07:05 2005", "notice", "done"]
    pub values: Vec<&'a str>,
    // Can't get this to compile. Doesn't seem to be a big deal perf-wise.
    // pub values: &'short[&'a str],
}

/// When a new log-template is discovered, rdrain will return this item.
/// Don't forget to refer to first_parse.
#[derive(Debug)]
pub struct NewTemplate<'a> {
    pub template: LogTemplate,
    pub first_parse: RecordParsed<'a>,
}

/// 
#[derive(Debug)]
pub enum RecordsParsedResult<'a> {
    NewTemplate(NewTemplate<'a>),
    RecordParsed(RecordParsed<'a>),
    ParseError(ParseError),
}

/// Iterator yielding every log record in the input string. A log record is generally a log-line,
/// but can be multi-line.
pub struct RecordsParsedIter<'a, 'b: 'a> {
    pub input: &'a str,
    pub state: &'b mut ParseTree,
    tokens: Vec<TokenParse<'a>>,
    parsed: &'a mut Vec<&'a str>,
}

impl<'a, 'b> RecordsParsedIter<'a, 'b> {
    pub fn from(input: &'a str, state: &'b mut ParseTree, parsed_buffer: &'a mut Vec<&'a str>) -> RecordsParsedIter<'a, 'b> {
        RecordsParsedIter {
            input,
            state,
            tokens: Vec::new(),
            parsed: parsed_buffer,
        }
    }
}

impl<'a, 'b> Iterator for RecordsParsedIter<'a, 'b> {
    type Item = RecordsParsedResult<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let split_result = self.input.split_once('\n');
        let (line, next_input) = match split_result {
            Some((line, rest)) => (line.strip_suffix('\r').unwrap_or(line), rest),
            None => (self.input, &self.input[0..0]),
        };
        self.input = next_input;
        if line.is_empty() {
            return None;
        }
        // TODO we should be able to handle multi-line logs, but the original paper doesn't.
        // It is easily fixed in the python by looking back, not so simple here.
        // We will probably have to iterate through lines looking ahead.


        // Step 1. First we split the line to get all of the tokens.
        // add_log_message from drain3.py
        // E.g. splits the line into chunks <timestamp> <loglevel> <content>
        let line_chunks = split_line_provided(line); // Pre-defined chunks as user-specified, like <Time> <Content>
        if line_chunks.is_none() {
            // Couldn't parse with the given regex, it's probably a multiline string. Attach it to the last-emitted log.
            // TODO attach_to_last_line(line);
            return Some(RecordsParsedResult::ParseError(
                ParseError::NoTokensInRecord,
            ));
        }
        // TODO Let Content be something not the last thing in the msg, like for trailing log-line tags.
        let line_chunks = line_chunks.unwrap();
        let log_content = *line_chunks.iter().rev().next().unwrap();

        // This is the masking feature from drain3; not clear what scenarios
        // would be best to use it, but running without it for now.
        let mut preprocessed = Vec::new();
        if false {
            let re = Regex::new(r"\d+").unwrap();
            let mut last_index = 0;
            for mmatch in re.find_iter(log_content) {
                if mmatch.start() > last_index {
                    preprocessed.push(Preprocessed::Segment(
                        &log_content[last_index..mmatch.start()],
                    ));
                }
                preprocessed.push(Preprocessed::Value(mmatch.as_str()));
                last_index = mmatch.end();
            }
            if last_index != log_content.len() {
                preprocessed.push(Preprocessed::Segment(&log_content[last_index..]));
            }
        } else {
            preprocessed.push(Preprocessed::Segment(log_content));
        }

        let tokens = &mut self.tokens;
        tokens.clear();
        debug!("preprocessed={:?}", preprocessed);
        for elem in preprocessed {
            match elem {
                Preprocessed::Segment(s) => tokens.extend(
                    s.split([' ', '\t'])
                        .filter(|s| !s.is_empty())
                        .map(TokenParse::Token),
                ),
                Preprocessed::Value(v) => tokens.push(TokenParse::MaskedValue(v)),
            }
        }

        if tokens.is_empty() {
            return Some(RecordsParsedResult::ParseError(ParseError::NoTokensInRecord));
        }

        // Step 2, we map #(num_tokens) => a parse tree with limited depth.
        let match_cluster = tree_search(&self.state.root, tokens);

        if match_cluster.is_none() {
            // We could also inline add_seq_to_prefix_tree here,
            // Either the prefix tree did not exist, in which case we have to add it and a one-cluster leaf-node.
            // Or, the prefix tree did exist, but no cluster matched above the threshold, so we need to add a cluster there.
            let match_cluster = Some(add_seq_to_prefix_tree(
                &mut self.state.root,
                &tokens,
                &mut self.state.next_cluster_id,
            )).unwrap();
            self.parsed.clear();
            zip_tokens_and_template(&match_cluster.template, &tokens, self.parsed);
            return Some(Self::Item::NewTemplate(NewTemplate {
                // We can't return this because it would imply that our mutable self borrow in Self::next outlives 'a.
                // We could make this less-copy by using a streaming-iterator or just taking callbacks to call.
                // Unclear which would be more idiomatic, so leaving it alone for now.
                template: match_cluster.template.to_vec(),
                first_parse: RecordParsed {
                    values: self.parsed.to_vec(),
                    template_id: match_cluster.cluster_id,
                },
            }));
        }
        let match_cluster = match_cluster.unwrap();
        debug!("Line {} matched cluster: {:?}", line, match_cluster);
        // It feels like it should be doable to pass tokens without collecting it first,
        // maintaining its lifetime as pointing to the original record. but skipped for now
        // since can't figure out how to do that without .collect().
        self.parsed.clear();
        zip_tokens_and_template(&match_cluster.template, &tokens, self.parsed);
        return Some(Self::Item::RecordParsed(RecordParsed {
            values: self.parsed.to_vec(),
            template_id: match_cluster.cluster_id,
        }));
    }
}
// First map using token length, then map based on tokens until maxDepth, then we've found it.
// If all digits token, replace with "*"
fn similar_sequence_score(seq1: &Vec<&str>, seq2: &Vec<&str>) -> usize {
    let mut sum: usize = 0;
    for (i, x) in seq1.iter().enumerate() {
        if i >= seq2.len() {
            break;
        }
        sum += (*x == seq2[i]) as usize;
    }
    sum / seq1.len()
}

fn has_numbers(s: &str) -> bool {
    s.chars().any(char::is_numeric)
}

fn split_line_provided(_line: &str) -> Option<Vec<&str>> {
    // TODO Copy regex from python, return list of spanning indicators.
    Some(vec![_line])
}

#[derive(Debug)]
struct LogCluster {
    template: LogTemplate,
    cluster_id: usize,
    // size: i64 from Python == num_matches. Why track this? Seems to be only for debugging.
}

fn sequence_distance(seq1: &[LogTemplateItem], seq2: &[TokenParse]) -> (f64, i64) {
    assert!(seq1.len() == seq2.len());
    if seq1.is_empty() {
        return (1.0, 0);
    }
    let mut sim_tokens: i64 = 0;
    let mut num_of_par = 0;
    for (token1, token2) in seq1.iter().zip(seq2.iter()) {
        match token1 {
            LogTemplateItem::Value => num_of_par += 1,
            LogTemplateItem::StaticToken(token1) => match token2 {
                TokenParse::Token(token2) => {
                    if token1 == token2 {
                        sim_tokens += 1
                    }
                }
                TokenParse::MaskedValue(_) => num_of_par += 1,
            },
        }
    }
    // Params don't match because this way we just skip params, and find the actually most-similar one, rather than letting
    // templates like "* * * " dominate.

    // let retVal = f64::from(simTokens) / f64::from(seq1.len());
    let ret_val = sim_tokens as f64 / seq1.len() as f64;
    (ret_val, num_of_par)
}

// const SIMILARITY_THRESHOLD: f64 = 0.7;
const SIMILARITY_THRESHOLD: f64 = 0.4;
fn fast_match<'a>(logclusts: &'a Vec<LogCluster>, tokens: &[TokenParse]) -> Option<&'a LogCluster> {
    // Sequence similarity search.
    let mut max_similarity = -1.0;
    let mut max_param_count = -1;
    let mut max_cluster = None;
    // The rewritten Python version introduced an 'include wildcard in matching' flag that I don't see any reason to set to False.
    // so, omitted here, back to original algo from paper.

    for log_clust in logclusts {
        let (cur_similarity, cur_num_params) = sequence_distance(&log_clust.template, tokens);
        if cur_similarity > max_similarity
            || (cur_similarity == max_similarity && cur_num_params > max_param_count)
        {
            max_similarity = cur_similarity;
            max_param_count = cur_num_params;
            max_cluster = Some(log_clust);
        }
    }

    if max_similarity >= SIMILARITY_THRESHOLD {
        max_cluster
    } else {
        None
    }
}

const MAX_DEPTH: usize = 4;
const MAX_CHILDREN: usize = 100;
fn add_seq_to_prefix_tree<'a>(
    root: &'a mut TreeRoot,
    tokens: &Vec<TokenParse>,
    num_clusters: &mut usize,
) -> &'a LogCluster {
    // Make sure there is a num_token => middle_node element.
    let clust_id = *num_clusters;
    *num_clusters += 1;
    debug!("Adding seq {} to tree: {:?}", clust_id, tokens);
    let token_count = tokens.len();
    assert!(token_count >= 2);
    let mut cur_node = root.entry(token_count).or_insert_with(|| {
        GraphNodeContents::MiddleNode(MiddleNode {
            child_d: FxHashMap::default(),
        })
    });

    let mut current_depth = 1;
    for token in tokens {
        let inserter = || {
            if current_depth == MAX_DEPTH - 1 || current_depth == token_count - 1 {
                GraphNodeContents::LeafNode(Vec::new())
            } else {
                GraphNodeContents::MiddleNode(MiddleNode {
                    child_d: FxHashMap::default(),
                })
            }
        };

        // trace!("token: {:?} node {:?}", token, cur_node);
        cur_node = match cur_node {
            GraphNodeContents::MiddleNode(middle) => {
                assert!(!(current_depth >= MAX_DEPTH || current_depth >= token_count));
                // if token not matched in this layer of existing tree.
                let num_children = middle.child_d.len();
                match token {
                    TokenParse::MaskedValue(v) => middle
                        .child_d
                        .entry(LogTemplateItem::Value)
                        .or_insert_with(inserter),
                    TokenParse::Token(token) => {
                        let perfect_match_key =
                            LogTemplateItem::StaticToken(token.to_string());
                        let found_node = middle.child_d.contains_key(&perfect_match_key);

                        // Double-lookup pleases the borrow-checker :shrug:
                        if found_node {
                            middle.child_d.get_mut(&perfect_match_key).unwrap()
                        } else {
                            // At first glance, skipping over '*' entries here is unintuitive. However, if we've made it to
                            // adding, then there was not a satisfactory match in the tree already. So we'll copy the original
                            // algo and make a new node even if there is already a star here, as long as no numbers.
                            // if self.parametrize_numeric_tokens
                            // If it's a numerical token, take the * path.
                            if has_numbers(token) || num_children >= MAX_CHILDREN {
                                middle
                                    .child_d
                                    .entry(LogTemplateItem::Value)
                                    .or_insert_with(inserter)
                            } else {
                                // It's not a numerical token, and there is room (maxChildren), add it.
                                middle
                                    .child_d
                                    .entry(perfect_match_key)
                                    .or_insert_with(inserter)
                            }
                        }
                    }
                }
            }
            GraphNodeContents::LeafNode(leaf) => {
                // if at max depth or this is last token in template - add current log cluster to the leaf node
                assert!(current_depth >= MAX_DEPTH || current_depth >= token_count);
                leaf.push(LogCluster {
                    template: tokens
                        .iter()
                        .map(|tp| match tp {
                            TokenParse::Token(t) => {
                                match has_numbers(t) {
                                    true => LogTemplateItem::Value,
                                    false => LogTemplateItem::StaticToken(t.to_string()),
                                }
                            }
                            TokenParse::MaskedValue(v) => LogTemplateItem::Value,
                        })
                        .collect(),
                    cluster_id: clust_id,
                });
                debug!("tree: {:?}", leaf);
                return &leaf[leaf.len() - 1];
            }
        };
        current_depth += 1
    }
    unreachable!();
}

// https://developer.ibm.com/blogs/how-mining-log-templates-can-help-ai-ops-in-cloud-scale-data-centers/

fn tree_search<'a>(root: &'a TreeRoot, tokens: &[TokenParse]) -> Option<&'a LogCluster> {
    let token_count = tokens.len();
    assert!(token_count != 0);
    let e = root.get(&token_count);
    // No template with same token count yet.
    e?;

    let mut cur_node = e.unwrap();
    /*if let GraphNodeContents::LeafNode(p) = parentn {
        unreachable!("Shouldn't be possible.");
    }*/
    // let GraphNodeContents::MiddleNode(mut parentn) = parentn;
    let mut current_depth = 1;
    for token in tokens {
        if current_depth >= MAX_DEPTH {
            break;
        }

        let middle = match cur_node {
            GraphNodeContents::MiddleNode(x) => x,
            GraphNodeContents::LeafNode(_) => {
                // Done, at leaf-node.
                assert!(current_depth == token_count);
                break;
            }
        };

        // If we know it's a Value, go ahead and take that branch.
        match token {
            TokenParse::MaskedValue(v) => {
                let maybe_next = middle.child_d.get(&LogTemplateItem::Value);
                if let Some(next) = maybe_next {
                    cur_node = next;
                } else {
                    return None;
                }
            }
            TokenParse::Token(token) => {
                // Actually walking to next child, look for the token, or a wildcard, or fail.
                let maybe_next = middle
                    .child_d
                    .get(&LogTemplateItem::StaticToken(token.to_string()));
                if let Some(next) = maybe_next {
                    cur_node = next;
                } else if let Some(wildcard) = middle.child_d.get(&LogTemplateItem::Value) {
                    cur_node = wildcard;
                } else {
                    return None; // Tried going down prefix tree that did not exist, need to make a new entry.
                }
            }
        }
        current_depth += 1;
    }
    // We have arrived at a list of LogClusters in a leaf-node.
    // Now, from these clusters, we need to pick the one that matches the closest.
    let log_clust = match cur_node {
        GraphNodeContents::MiddleNode(_) => unreachable!("Mistake."),
        GraphNodeContents::LeafNode(x) => x,
    };
    let ret_log_clust = fast_match(log_clust, tokens);
    ret_log_clust
}

#[derive(Debug)]
struct MiddleNode {
    child_d: FxHashMap<LogTemplateItem, GraphNodeContents>,
}

#[derive(Debug)]
enum GraphNodeContents {
    MiddleNode(MiddleNode),
    LeafNode(Vec<LogCluster>),
}

type TreeRoot = FxHashMap<usize, GraphNodeContents>;
pub type LogTemplate = Vec<LogTemplateItem>;

use regex::{Match, Regex, Split};

use std::fs::{File, read_to_string};
use std::io::{self, prelude::*, BufReader};

pub fn print_log(filename: &str) {
    // TODO Make a CLI version
    let s: _ = read_to_string(filename).unwrap();
    let mut state = ParseTree::default();
    let mut template_names = Vec::new();
    for record in RecordsParsedIter::from(&s, &mut state, &mut Vec::new()) {
        fn handle_parse(template_names: &[String], rp: &RecordParsed) {
            let typ = &template_names[rp.template_id];
            let obj  = json_object!{
                template: typ,
                values: ToJSONList(rp.values.to_vec())};
            info!("json: {}", obj.to_json_string());
        }

        match record {
            RecordsParsedResult::NewTemplate(template) => {
                template_names.push(
                    template.template
                    .iter()
                    .map(|t| t.to_string())
                    .intersperse(" ".to_string())
                    .collect::<String>());

                    handle_parse(&template_names, &template.first_parse);
            }
            crate::RecordsParsedResult::RecordParsed(rp) => handle_parse(&template_names, &rp),
            crate::RecordsParsedResult::ParseError(e) => error!("err: {}", e),
        }
    }
}
