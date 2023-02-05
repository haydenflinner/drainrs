#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#[allow(unused_imports)]
use indextree::Arena;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BTreeMap, HashSet};
use std::{collections::HashMap, error::Error};

use log::{debug, info, warn};

// type inference lets us omit an explicit type signature (which
// would be `BTreeMap<&str, &str>` in this example).
fn btree() {
    let mut movie_reviews = BTreeMap::new();

    // review some movies.
    movie_reviews.insert("Office Space", "Deals with real issues in the workplace.");
    movie_reviews.insert("Pulp Fiction", "Masterpiece.");
    movie_reviews.insert("The Godfather", "Very enjoyable.");
    movie_reviews.insert("The Blues Brothers", "Eye lyked it a lot.");
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

#[derive(Debug)]
struct LogCluster {
    template: Vec<String>,
    cluster_id: i64,
    // size: i64 from Python == num_matches. Why track this? Seems to be only for debugging.
}

fn sequence_distance(seq1: &Vec<String>, seq2: &Vec<&str>) -> (f64, i64) {
    assert!(seq1.len() == seq2.len());
    if seq1.len() == 0 {
        return (1.0, 0);
    }
    let mut sim_tokens: i64 = 0;
    let mut num_of_par = 0;
    for (token1, token2) in seq1.iter().zip(seq2.iter()) {
        if *token1 == "<*>" {
            num_of_par += 1;
            continue;
        }
        if token1 == token2 {
            sim_tokens += 1;
        }
    }
    // Params don't match because this way we just skip params, and find the actually most-similar one, rather than letting
    // templates like "* * * " dominate.

    // let retVal = f64::from(simTokens) / f64::from(seq1.len());
    let ret_val = sim_tokens as f64 / seq1.len() as f64;
    return (ret_val, num_of_par);
}

const SIMILARITY_THRESHOLD: f64 = 0.7;
fn fast_match<'a>(logclusts: &'a Vec<LogCluster>, tokens: &Vec<&str>) -> Option<&'a LogCluster> {
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
    tokens: &Vec<&str>,
    num_clusters: &mut i64,
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
        debug!("token: {} node {:?}", token, cur_node);
        cur_node = match cur_node {
            GraphNodeContents::MiddleNode(middle) => {
                assert!(!(current_depth >= MAX_DEPTH || current_depth >= token_count));
                // if token not matched in this layer of existing tree.
                let num_children = middle.child_d.len();
                let found_node = middle.child_d.contains_key(&token.to_string());

                // Double-lookup pleases the borrow-checker :shrug:
                if found_node {
                    middle.child_d.get_mut(&token.to_string()).unwrap()
                } else {
                     // At first glance, skipping over '*' entries here is unintuitive. However, if we've made it to
                    // adding, then there was not a satisfactory match in the tree already. So we'll copy the original
                    // algo and make a new node even if there is already a star here, as long as no numbers.
                    // if self.parametrize_numeric_tokens
                    // If it's a numerical token, take the * path.
                    if has_numbers(token) {
                        middle.child_d.entry("<*>".to_string()).or_insert_with(|| {
                            if current_depth == MAX_DEPTH - 1 || current_depth == token_count - 1 {
                                GraphNodeContents::LeafNode(Vec::new())
                            } else {
                                GraphNodeContents::MiddleNode(MiddleNode {
                                    child_d: HashMap::new(),
                                })
                            }
                        })
                    } else {
                        // It's not a numerical token, so if there's room (maxChildren), add it.
                        // Finally, ensure that there is a wildcard option and take it.
                        let key = if num_children < MAX_CHILDREN {
                            token
                        } else {
                            "<*>"
                        };
                        middle.child_d.entry(key.to_string()).or_insert_with(|| {
                            if current_depth == MAX_DEPTH - 1 || current_depth == token_count - 1 {
                                GraphNodeContents::LeafNode(Vec::new())
                            } else {
                                GraphNodeContents::MiddleNode(MiddleNode {
                                    child_d: HashMap::new(),
                                })
                            }
                        })
                    }
                }
            }
            GraphNodeContents::LeafNode(leaf) => {
                // if at max depth or this is last token in template - add current log cluster to the leaf node
                assert!(current_depth >= MAX_DEPTH || current_depth >= token_count);
                leaf.push(LogCluster {
                    template: tokens.iter().map(|s| String::from(*s)).collect(),
                    cluster_id: *num_clusters,
                });
                return &leaf[leaf.len() - 1];
                // return;
            }
        };
        current_depth += 1
    }
    unreachable!();
}

/*macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}*/

fn tree_search<'a>(root: &'a TreeRoot, tokens: &Vec<&str>) -> Option<&'a LogCluster> {
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

        // Actually walking to next child, look for the token or a wildcard.
        let maybe_next = middle.child_d.get(&token.to_string());
        if maybe_next.is_some() {
            cur_node = maybe_next.unwrap();
        } else if let Some(wildcard) = middle.child_d.get("<*>") {
            cur_node = wildcard;
        } else {
            return None; // Tried going down prefix tree that did not exist, need to make a new entry.
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
    child_d: HashMap<String, GraphNodeContents>,
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

use regex::{Match, Regex};
fn preprocess_domain_knowledge(s: &str) -> String {
    // TODO Configurable domain knowledge regexes, called "reg" in python
    // Worked fine for the research paper so will work fine here prolly.
    let re = Regex::new(r"\d+").unwrap();
    re.replace_all(s, "<*>").to_string()
}

use std::fs::File;
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
    let mut root: TreeRoot = TreeRoot::new();
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

    let mut cluster_id = 0;

    for line in reader.lines() {
        // Step 1. First we split the line to get all of the tokens.
        // add_log_message from drain3.py
        let line = line?;
        let line_chunks = split_line_provided(&line); // Pre-defined chunks as user-specified, like <Time> <Content>
        if let None = line_chunks {
            // Couldn't parse with the given regex, it's probably a multiline string. Attach it to the last-emitted log.
            // TODO attach_to_last_line(line);
            debug!("Skipping record that couldn't parse with user defined template, multiline log msg?");
            continue;
        }
        // TODO Let Content be something not the last thing in the msg.
        // For right now it's fine.
        let line_chunks = line_chunks.unwrap();
        let log_content = line_chunks.iter().rev().next().unwrap();

        // TODO It owuld be better to keep the string split apart into tokens, rather than rejoining to a string with <*>
        // both for runtime and for safety (what if <*> occurred in the log msg?)
        let masked = preprocess_domain_knowledge(log_content);
        let tokens: Vec<&str> = masked.split([' ', '\t']).collect();
        if tokens.len() == 0 {
            unimplemented!("Empty log line or empty parsed. Can't just let this slide, can cause issues in many places..what to do?");
        }

        // Step 2, we map #(num_tokens) => a parse tree with limited depth.
        let mut match_cluster = tree_search(&root, &tokens);

        if match_cluster.is_none() {
            // We could also inline add_seq_to_prefix_tree here,
            // Either the prefix tree did not exist, in which case we have to add it and a one-cluster leaf-node.
            // Or, the prefix tree did exist, but no cluster matched above the threshold, so we need to add a cluster there.
            match_cluster = Some(add_seq_to_prefix_tree(&mut root, &tokens, &mut cluster_id));
        } else {
            info!("Matched cluster.");
        }
        let c = match_cluster.unwrap();
        info!("Line {} matched cluster: {:?}", line, c);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    // use super::*;

    use crate::parse_emit_csv;

    #[test]
    fn test_add() {
        assert_eq!((1 + 2), 3);
    }

    /*
    #[test]
    fn test_bad_add() {
        // This assert would fire and test will fail.
        // Please note, that private functions can be tested too!
        assert_eq!((1 - 2), 3);
    }
    */
    
    use simple_logger::SimpleLogger;
    #[test]
    fn test_a() {
        SimpleLogger::new().init().unwrap();
        parse_emit_csv("testlog1.txt");
        assert_eq!(1, 0);
    }
}
