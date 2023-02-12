#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![feature(iter_intersperse)]
use indextree::Arena;
use simple_logger::SimpleLogger;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::{collections::HashMap, error::Error};
use thiserror::Error;
use serde_json::json;
use log::{debug, info, warn};

/// In the process of parsing, the drain algo creates a Parse tree. This tree can be saved
/// and re-used on the next run, to avoid "forgetting" the previously recognized log templates.
#[derive(Default)]
pub struct ParserState {
    root: TreeRoot,
    next_cluster_id: usize,
    // = TreeRoot::new();
}
use std::iter::zip;

// Holy cow what ugly syntax to make this generic lol:
// https://www.reddit.com/r/learnrust/comments/t18ue0/comment/hyepyn9/?utm_source=share&utm_medium=web2x&context=3
// fn zip_tokens_and_template<'a, 'b>(templatetokens: &[String], logtokens: &'b[&'a str] ) -> Vec<&'a str> {
// fn zip_tokens_and_template<'a, 'b>(templatetokens: &[String], logtokens: &'b[&'a str] ) -> Vec<&'b str> {
fn zip_tokens_and_template<'a, 'b, 'c>(
    templatetokens: &'a [OwningLogTemplateItem],
    logtokens: &'b [TokenParse<'c>],
) -> Vec<&'c str> {
    let mut results = Vec::new();
    for (template_token, log_token) in zip(templatetokens, logtokens) {
        match template_token {
            OwningLogTemplateItem::StaticToken(_) => {}
            OwningLogTemplateItem::Value => match log_token {
                TokenParse::Token(v) => results.push(*v),
                TokenParse::MaskedValue(v) => results.push(*v),
            },
        }
    }
    results
}

/*struct NewTemplate<'a> {
    line: &'a str,

}*/
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum OwningLogTemplateItem {
    StaticToken(String), // Owned because we need to store it.
    Value,               // Python port used "<*>" instead.
}

impl fmt::Display for OwningLogTemplateItem {
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

/*impl From<TokenParse<'a>> for OwningLogTemplateItem {
    fn from(tp: TokenParse<'a>) -> Self {
        match ()
        OwningLogTemplateItem::StaticToken(())
    }

}*/

/*pub enum LogTemplateItem<'a> {
    StaticToken(&'a str),
    Value,
    I can't convince the borrow checker today to allow this :-)
}*/
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

#[derive(Debug)]
pub struct RecordParsed<'a> {
    pub template_id: usize,
    // I'd prefer somehow returning an undefined size iterator over a vec,
    // but we're already in one iterator, perf probably is not that important lol.
    pub values: Vec<&'a str>,
    // values: Vec<^'a str>,
}

#[derive(Debug)]
pub struct NewTemplate<'a> {
    pub template: Vec<OwningLogTemplateItem>,
    pub first_parse: RecordParsed<'a>,
}

#[derive(Debug)]
pub enum RecordsParsedResult<'a> {
    NewTemplate(NewTemplate<'a>),
    RecordParsed(RecordParsed<'a>),
    ParseError(ParseError),
}

/// Iterator yielding every log record in the input string. A log record is generally a log-line,
/// but can be multi-line.
pub struct RecordsParsed<'a> {
    pub input: &'a str,
    pub state: &'a mut ParserState,
}

impl<'a> RecordsParsed<'a> {
    pub fn from(input: &'a str, state: &'a mut ParserState) -> RecordsParsed<'a> {
        RecordsParsed {
            input: input,
            state: state,
        }
    }
}

impl<'a> Iterator for RecordsParsed<'a> {
    type Item = RecordsParsedResult<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        info!("self.input={:?}", self.input);
        let split_result = self.input.split_once('\n');
        info!("split_result={:?}", split_result);
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
        let line_chunks = split_line_provided(&line); // Pre-defined chunks as user-specified, like <Time> <Content>
        if line_chunks.is_none() {
            // Couldn't parse with the given regex, it's probably a multiline string. Attach it to the last-emitted log.
            // TODO attach_to_last_line(line);
            return Some(RecordsParsedResult::ParseError(
                ParseError::NoTokensInRecord,
            ));
        }
        // TODO Let Content be something not the last thing in the msg.
        let line_chunks = line_chunks.unwrap();
        let log_content = *line_chunks.iter().rev().next().unwrap();

        // TODO It owuld be better to keep the string split apart into tokens, rather than rejoining to a string with <*>
        // both for runtime and for safety (what if <*> occurred in the log msg?)
        // preprocess_domain_knowledge. I.e. splitting the content into strs and KnownValues.
        // We do this _before_ we split on whitespace in case the known value includes whitespace.
        // Example use: recognizing MyLoggedMsg{key: value} as one big token for automated parsing.
        // let re = Regex::new(r"\d+").unwrap();
        // This is the masking feature from drain3; not clear what scenarios
        // would be best to use it, but running without it for now.
        let mut preprocessed = Vec::new();
        if false {
            let re = Regex::new(r"\d+").unwrap();
            // let mut content_tokens = Vec::new();
            let mut last_index = 0;
            //for segment in re.split(log_content) {
            //if segment.ind
            //}
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
            preprocessed.push(Preprocessed::Segment(&log_content[..]));
        }

        // re.find_iter(log_content).map(|mmatch| PreProcessed(PreProcessed::Segment(mmatch.range()))).intersperse(separator)
        // let masked = {
        // TODO Configurable domain knowledge regexes, called "reg" in python
        // Worked fine for the research paper so will work fine here prolly.
        // re.split(log_content)
        // re.find_iter(text)
        //re.replace_all(s, "<*>").to_string()
        // };

        let mut tokens = Vec::new();
        debug!("preprocessed={:?}", preprocessed);
        for elem in preprocessed {
            match elem {
                Preprocessed::Segment(s) => tokens.extend(
                    s.split([' ', '\t'])
                        .filter(|s| s.len() > 0)
                        .map(|t| TokenParse::Token(t)),
                ),
                Preprocessed::Value(v) => tokens.push(TokenParse::MaskedValue(v)),
            }
        }

        if tokens.len() == 0 {
            unimplemented!("Empty log line or empty parsed. Can't just let this slide, can cause issues in many places..what to do?");
        }

        // Step 2, we map #(num_tokens) => a parse tree with limited depth.
        let match_cluster = tree_search(&self.state.root, &tokens);

        if match_cluster.is_none() {
            // We could also inline add_seq_to_prefix_tree here,
            // Either the prefix tree did not exist, in which case we have to add it and a one-cluster leaf-node.
            // Or, the prefix tree did exist, but no cluster matched above the threshold, so we need to add a cluster there.
            let match_cluster = Some(add_seq_to_prefix_tree(
                &mut self.state.root,
                &tokens,
                &mut self.state.next_cluster_id,
            ))
            .unwrap();
            return Some(Self::Item::NewTemplate(NewTemplate {
                template: match_cluster.template.to_vec(),
                first_parse: RecordParsed {
                    values: zip_tokens_and_template(&match_cluster.template, &tokens),
                    template_id: match_cluster.cluster_id,
                },
            }));
        }
        let match_cluster = match_cluster.unwrap();
        info!("Line {} matched cluster: {:?}", line, match_cluster);
        // It feels like it should be doable to pass tokens without collecting it first,
        // maintaining its lifetime as pointing to the original record. but skipped for now
        // since can't figure out how to do that without .collect().
        return Some(Self::Item::RecordParsed(RecordParsed {
            values: zip_tokens_and_template(&match_cluster.template, &tokens),
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

// SmallVec would be good here.
fn split_line_provided(_line: &str) -> Option<Vec<&str>> {
    // TODO Copy regex from python, return list of spanning indicators.
    // None
    let mut vec = Vec::new();
    vec.push(_line);
    Some(vec)
}

// fn get_template(new_template_tokens: Vec<&str>, old_template_tokens: Vec<&str>) -> Vec<&str> {
//     assert_eq!(new_template_tokens.len(), old_template_tokens.len());
//
//     new_template_tokens.iter()
//       .zip(old_template_tokens.iter())
//       .map(|(o, n)| if **o != **n { "<*>" } else {*o }).collect()
// }
//type Template = Vec<String>;
//struct

// TODO what should we do about quoted values with spaces in them?
// the naive tokenizer won't handle this.

// info!("hello this {} value goes here", 42)
// -> [ StaticToken("hello"), StaticToken("this"), Value("42"), StaticToken... ]
// Todo use this type-safe approach everywhere rather than the direct <*> port from Python.
#[derive(Debug)]
struct LogCluster {
    template: Vec<OwningLogTemplateItem>,
    cluster_id: usize,
    // size: i64 from Python == num_matches. Why track this? Seems to be only for debugging.
}

fn sequence_distance(seq1: &Vec<OwningLogTemplateItem>, seq2: &[TokenParse]) -> (f64, i64) {
    assert!(seq1.len() == seq2.len());
    if seq1.len() == 0 {
        return (1.0, 0);
    }
    let mut sim_tokens: i64 = 0;
    let mut num_of_par = 0;
    for (token1, token2) in seq1.iter().zip(seq2.iter()) {
        match token1 {
            OwningLogTemplateItem::Value => num_of_par += 1,
            OwningLogTemplateItem::StaticToken(token1) => match token2 {
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
    return (ret_val, num_of_par);
}

const SIMILARITY_THRESHOLD: f64 = 0.7;
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

// 20221213 Note; the plan is basically just to translate the Python to Rust
// And then build some tests
// Then we can refactor to our hearts content.
const MAX_DEPTH: usize = 100;
const MAX_CHILDREN: usize = 100;
fn add_seq_to_prefix_tree<'a>(
    root: &'a mut TreeRoot,
    tokens: &Vec<TokenParse>,
    num_clusters: &mut usize,
) -> &'a LogCluster {
    // Make sure there is a num_token => middle_node element.
    let clust_id = *num_clusters;
    *num_clusters += 1;
    info!("Adding seq {} to tree: {:?}", clust_id, tokens);
    let token_count = tokens.len();
    assert!(token_count >= 2);
    let mut cur_node = root.entry(token_count).or_insert_with(|| {
        GraphNodeContents::MiddleNode(MiddleNode {
            child_d: HashMap::new(),
        })
    });

    // TODO Could fix all this one-token nonsense by hardcoding a rule for 1-token "<*>".
    // 1 => { "ABC": [1 LogCLuster], "DEF": [1 LogCluster]}
    // 2 => {"ABC": { "token2": [ x logclusters...]}}

    let mut current_depth = 1;
    for token in tokens {
        let inserter = || {
            if current_depth == MAX_DEPTH - 1 || current_depth == token_count - 1 {
                GraphNodeContents::LeafNode(Vec::new())
            } else {
                GraphNodeContents::MiddleNode(MiddleNode {
                    child_d: HashMap::new(),
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
                        .entry(OwningLogTemplateItem::Value)
                        .or_insert_with(inserter),
                    TokenParse::Token(token) => {
                        let perfect_match_key =
                            OwningLogTemplateItem::StaticToken(token.to_string());
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
                                    .entry(OwningLogTemplateItem::Value)
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
                                    true => OwningLogTemplateItem::Value,
                                    false => OwningLogTemplateItem::StaticToken(t.to_string()),
                                }
                            }
                            TokenParse::MaskedValue(v) => OwningLogTemplateItem::Value,
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
    if e.is_none() {
        return None;
    }

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
                let maybe_next = middle.child_d.get(&OwningLogTemplateItem::Value);
                if maybe_next.is_some() {
                    cur_node = maybe_next.unwrap();
                } else {
                    return None;
                }
            }
            TokenParse::Token(token) => {
                // Actually walking to next child, look for the token, or a wildcard, or fail.
                let maybe_next = middle
                    .child_d
                    .get(&OwningLogTemplateItem::StaticToken(token.to_string()));
                if maybe_next.is_some() {
                    cur_node = maybe_next.unwrap();
                } else if let Some(wildcard) = middle.child_d.get(&OwningLogTemplateItem::Value) {
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
    return ret_log_clust;
}

//enum GraphNodeContents {
// NumTokens(usize), // First-level
// Token(String), // Intermediate-levels
// I know the paper diagram shows a List of LogGroups as the leaf node,
// but what they actually do in the code is store one LogEvent which contains a template,
// And they just update that template to parameterize values which are later seen to have changed.
// I'm not storing logIDs because that was apparently only for the later post analysis in the python.
// Last level, leaf, contains log template.
// The paper also says that it will traverse "depth" nodes in Step3, where depth is the amxDepth.
// This isn't strictly true, there are _at most_ that many nodes.
// LogCluster(Vec<String>)
// LogCluster(Vec<String>)
// }

// TODO Replace this with the probably-much-faster vec-arena-graph thing first drafted.
// indextree and o(n) children walk through doing equality, rather than hashing? let's leave it alone til it works then we can make it fast.
#[derive(Debug)]
struct MiddleNode {
    child_d: HashMap<OwningLogTemplateItem, GraphNodeContents>,
    // TODO Remove The value of token here is not clear since it is also known by the path that we walked to the node from?
    // token: String,
}

#[derive(Debug)]
enum GraphNodeContents {
    MiddleNode(MiddleNode),
    LeafNode(Vec<LogCluster>),
}

// type TreeNode = HashMap<String, GraphNodeContents>;
type TreeRoot = HashMap<usize, GraphNodeContents>;

use regex::{Match, Regex, Split};

use std::fs::{File, read_to_string};
#[allow(unused_imports)]
use std::io::{self, prelude::*, BufReader};
fn parse_emit_csv(filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // let mut root: BTreeMap<usize, BTreeMap<String, String>>;
    // let mut last_line: Option<String>;

    // let mut arena: Arena<String> = Arena::new();
    // let arena: mut Arena<String>;
    // let a = arena.new_node(GraphNodeContents::Token("xyz".to_string()));
    // let root = arena.new_node(GraphNodeContents::NumTokens(0));
    // root.prepend(new_child, arena)
    // root.insert_after(new_sibling, arena)
    // arena.nodes[root.index].first_child

    // With input regex <timestamp> <level> <content>
    // Let's start with a log line "123 INFO mytokens go here now"
    // First we group with regexes, getting something like:
    // [timestamp, info, content]
    // Then we take line['content] and split.
    // logMessageL = line['content']
    // We're basically bypassing the tree with our hand-written regex.

    // let mut cluster_id = 0;

    /*for line in reader.lines() {
    }*/
    todo!("Use the new API here.");
    // Ok(())
}

pub fn print_log(filename: &str) {
    SimpleLogger::new().init().unwrap();
    //let file = File::open("testlog1.txt").unwrap();
    //let reader = BufReader::new(file);
    // parse_emit_csv("testlog1.txt");
    // TODO Make a CLI version
    // TODO emit results as a template plus a dictionary?
    // template as str minus wildcards?
    // let s: _ = read_to_string("testlog1.txt").unwrap();
    // let s: _ = read_to_string("Apache.log").unwrap();
    let s: _ = read_to_string(filename).unwrap();
    let mut state = ParserState::default();
    // let mut tables = Vec::<Vec::<Vec::<&str>>>::new();
    // Let's convert to jsonl.
    // let mut tables: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut template_names = Vec::new();
    for record in (RecordsParsed {
        input: &s,
        state: &mut state,
    }) {
        fn handle_parse(template_names: &Vec<String>, rp: &RecordParsed) {
            // let m: HashMap<String, String> = HashMap::new();
            // prefixes=
            //let obj = object!["type": typ];
            let typ = &template_names[rp.template_id];
            let obj = json!({
                "template": typ,
                "values": rp.values});
            //obj.insert(typ, rp.values);
            info!("json: {}", obj.to_string());
            // info!("user received: {:?}", rp);
            // &template_names[rp.template_id];// .push(rp.values);
        }

        match record {
            RecordsParsedResult::NewTemplate(template) => {
                template_names.push(
                    template.template
                    .iter()
                    .map(|t| t.to_string())
                    .intersperse(" ".to_string())
                        // .join(", ")
                    .collect::<String>());

                    handle_parse(&template_names, &template.first_parse);
            }
            crate::RecordsParsedResult::RecordParsed(rp) => handle_parse(&template_names, &rp),
            crate::RecordsParsedResult::ParseError(_) => unimplemented!(),
        }
    }
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    // use super::*;

    use std::{
        collections::HashMap,
        fs::{read_to_string, File},
        io::BufReader,
    };

    use crate::{parse_emit_csv, ParserState, RecordsParsed, RecordParsed};

    #[test]
    fn test_add() {
        assert_eq!((1 + 2), 3);
    }

    use log::info;
    /*
    #[test]
    fn test_bad_add() {
        // This assert would fire and test will fail.
        // Please note, that private functions can be tested too!
        assert_eq!((1 - 2), 3);
    }
    */

}
