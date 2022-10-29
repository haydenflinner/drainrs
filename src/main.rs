fn main() {
    println!("Hello, world!");
}

#[allow(unused_imports)]
use std::collections::{BTreeMap, HashSet};
use std::error::Error;

// type inference lets us omit an explicit type signature (which
// would be `BTreeMap<&str, &str>` in this example).
fn btree() {
let mut movie_reviews = BTreeMap::new();

// review some movies.
movie_reviews.insert("Office Space",       "Deals with real issues in the workplace.");
movie_reviews.insert("Pulp Fiction",       "Masterpiece.");
movie_reviews.insert("The Godfather",      "Very enjoyable.");
movie_reviews.insert("The Blues Brothers", "Eye lyked it a lot.");
}

// First map using token length, then map based on tokens until maxDepth, then we've found it.
// If all digits token, replace with "*"
fn similar_sequence_score(seq1: &Vec<&str>, seq2: &Vec<&str>) -> usize {
    let mut sum: usize = 0;
    for (i, x) in seq1.iter().enumerate() {
        if i >= seq2.len() { break; }
        sum += (*x == seq2[i]) as usize;
    }
    sum / seq1.len()
}

// SmallVec would be good here.
fn split_line_provided(_line: &str) -> Option<Vec<&str>> {
    // TODO Copy regex from python, return list of spanning indicators.
    None
}

/*
// I'm sure there are good crates for this but I found this easy to copy paste from leipzig's page
// and since we're never removing nodes it's gonna be hard to beat a flat vector I think. We really
// shouldn't be using too much memory.
pub struct Arena<T> {
    nodes: Vec<Node<T>>,
}
pub struct Node<T> {
    // parent: Option<NodeId>,

    previous_sibling: Option<NodeId>,
    next_sibling: Option<NodeId>,
    first_child: Option<NodeId>,
    last_child: Option<NodeId>,

    /// The actual data which will be stored within the tree
    pub data: T,
}

pub struct NodeId {
    index: usize,
}

impl<T> Arena<T> {
    fn new() -> Arena<T> { Arena{nodes: Vec::new()} }

    pub fn new_node(&mut self, data: T) -> NodeId {
    // Get the next free index
    let next_index = self.nodes.len();

    // Push the node into the arena
    self.nodes.push(Node {
        // parent: None,
        first_child: None,
        last_child: None,
        previous_sibling: None,
        next_sibling: None,

        data: data,
    });

    // Return the node identifier
    NodeId { index: next_index }
}
}
*/

fn get_template(new_template_tokens: Vec<&str>, old_template_tokens: Vec<&str>) -> Vec<&str> {
    assert_eq!(new_template_tokens.len(), old_template_tokens.len());

    new_template_tokens.iter()
      .zip(old_template_tokens.iter())
      .map(|(o, n)| if **o != **n { "<*>" } else {*o }).collect()
}

struct MatchCluster {
    HashMap<String, String> map;
}

fn fast_match(logclust : &mut MatchCluster, tokens : Vec<&str>) -> Option<MatchCluster> {
  // Sequence similarity search.
}

// TODO this looks complicated lol
fn add_seq_to_prefix_tree() {}

enum GraphNodeContents {
    NumTokens(usize), // First-level
    Token(String), // Intermediate-levels
    // I know the paper diagram shows a List of LogGroups as the leaf node,
    // but what they actually do in the code is store one LogEvent which contains a template,
    // And they just update that template to parameterize values which are later seen to have changed.
    // I'm not storing logIDs because that was apparently only for the later post analysis in the python.
    // Last level, leaf, contains log template.
    // The paper also says that it will traverse "depth" nodes in Step3, where depth is the amxDepth.
    // This isn't strictly true, there are _at most_ that many nodes.
    MatchCluster(Vec<String>)
    MatchCluster(Vec<String>)
}

use indextree::Arena;

/*
fn tree_search(arena: &Arena<String>, start: &NodeId, seq: &Vec<&str>) -> Node<String> {
    let mut idx = start.index;
    let mut depth = 1;
    let max_depth = 100;

    let mut cur_node = &arena.nodes[0];
    for token in seq {
        if depth >= 100 /*|| depth > seq.len()*/ { break; }
        let node = &arena.nodes[idx];
        // if node == 
    }
    arena.nodes[start.index].data
    // arena.nodes[start.index]
}
*/

use regex::Regex;
fn preprocess_domain_knowledge(s: &str) -> String {
    // TODO Configurable domain knowledge regexes, called "reg" in python
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

    let mut arena = Arena::new();
    // let arena: mut Arena<String>;
    // let a = arena.new_node(GraphNodeContents::Token("xyz".to_string()));
    let root = arena.new_node(GraphNodeContents::NumTokens(0));
    // root.prepend(new_child, arena)
    // root.insert_after(new_sibling, arena)
    // arena.nodes[root.index].first_child

    for line in reader.lines() {
        // Step 1. First we split the line to get all of the tokens.
        let line = line?;
        let line_chunks = split_line_provided(&line); // Pre-defined chunks as user-specified, like <Time> <Content>
        if let None = line_chunks {
            // Couldn't parse with the given regex, it's probably a multiline string. Attach it to the last-emitted log.
            // TODO attach_to_last_line(line);
            continue;
        }
        // TODO Let Content be something not the last thing in the msg.
        let log_content = line_chunks.unwrap().iter().rev().next().unwrap();
        // TODO It owuld be better to keep the string split apart into tokens, rather than rejoining to a string with <*>
        // both for runtime and for safety (what if <*> occurred in the log msg?)
        let tokens: Vec<&str> = preprocess_domain_knowledge(log_content).split([' ', '\t']).collect();
        // let tokens = tokens.unwrap();
        // Step 2, we map #(num_tokens) => a parse tree with limited depth.
        let len = tokens.len();
        let maybe_matching_num_node = root.reverse_children(&arena).find_map(|n| {
            match arena.get(n)?.get() {
                GraphNodeContents::NumTokens(node_tokens) => {
                    if *node_tokens == len { return Some(n); }
                    return None;
                }
                _ => { None }
            }
        });
        // Have iterated through all of the number nodes, did we find one or do we need to make one?
        // TODO Check if python respects maxChildren here, I doubt it.
        let num_node = match maybe_matching_num_node {
            Some(id) => id,
            None => {
                let new_node_id = arena.new_node(GraphNodeContents::NumTokens(len));
                root.append(new_node_id, &mut arena);
                new_node_id
            }
        };
        // Now we start to search the children of the number-node. Each node may have up to maxChildren direct children.
        // this is the part that might benefit from some sort of HashSet, so we don't grow to O(n) the number of children.
        // It does use a hashset in the Python impl.
        let mut cur_token_iter = tokens.iter();
        let mut cur_node_iter = num_node.reverse_children(&arena);
        let mut prev_node = num_node;
        let mut cur_node = cur_node_iter.next();
        let mut cur_token = cur_token_iter.next();
        // let max_children = 100;
        let max_depth = 100;
        let mut depth = 2;
        loop {
            // let cur_token = cur_token_iter.as_ref();
            // let cur_token = cur_token_iter.next();
            // let cur_node = cur_node_iter.next();
            // TODO in two branches of this, replace token containing digits with "*".
            match (cur_token, cur_node) {
            // match cur_node_iter.next() {
                // I guess either of these are possible if we start with an empty line after preprocessing.
                // But they shouldn't happen once we've iterated once, that would indicate algo error here.
                (None, None) => unreachable!("No next token or node."),
                (None, Some(_)) => unreachable!("No next token but have a node."),

                (Some(token), None) => {
                    // We don't have a match for the current token. need to add one to the parent.
                    // Unless we've exhausted the sequence 
                    let new_node;
                    if depth == max_depth || cur_token_iter.peekable().next().is_none() {
                        // Time to append a non-token leaf-node.
                        // TODO Here we break the statement in the paper that all leaf nodes
                        // are at the same depth. Problem is, it doesn't appear to be true in the Python either.
                        // This maps most closely to treeSearch returning a matchCluster.
                        let mut templates = HashSet::new();

                        // TODO Handle creating template, i.e. "created xyz 123" -> "created * *"
                        templates.insert("LOL".to_string());
                        new_node = arena.new_node(GraphNodeContents::MatchCluster(templates));
                        prev_node.append(new_node, &mut arena);
                        break;
                    } else {
                        new_node = arena.new_node(GraphNodeContents::Token(token.to_string()));
                        prev_node.append(new_node, &mut arena);
                        prev_node = new_node;
                        cur_node_iter = new_node.reverse_children(&arena); // Rather than special-case appending rest to the tree.
                        depth += 1;
                        cur_token = cur_token_iter.next();
                        cur_node = cur_node_iter.next();
                    }
                },
                (Some(token), Some(node_id)) => {
                    // match on type of node. May be either a token node or a log group node.
                    match arena.get(node_id).expect("node never deleted").get() {
                        GraphNodeContents::Token(node_token) => {
                            if node_token == token {
                                cur_node_iter = node_id.reverse_children(&arena); // Rather than special-case appending rest to the tree.
                                depth += 1;
                                cur_token = cur_token_iter.next();
                                cur_node = cur_node_iter.next();
                            } else { // No match, look at next sibling in tree.
                                cur_node = cur_node_iter.next();
                            }
                        },
                        // We have reached a leaf. Check if we match any of the already created templates.
                        // If so, we have a match. If not, make a new entry.
                        GraphNodeContents::Templates(_) => todo!(),

                        GraphNodeContents::NumTokens(_) => unreachable!(),
                    }
                },
            }
            /*
            match arena.get(n)?.get() {
                GraphNodeContents::Token(token) => {
                    if *token == *cur_token {

                    }
                }
            }
            */
        }
        /*num_node.reverse_children(&arena).find_map(|n| {
        });*/
        // arena[root].ar
        // if !root.contains_key(&len) { root.insert(len, Arena::new()); }
        // let tree = root[&tokens.len()];
    }

    Ok(())
}

/*
#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    // use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 3);
    }

    #[test]
    fn test_bad_add() {
        // This assert would fire and test will fail.
        // Please note, that private functions can be tested too!
        assert_eq!(bad_add(1, 2), 3);
    }
}
*/