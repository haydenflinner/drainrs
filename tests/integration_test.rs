#![feature(iter_intersperse)]

use std::fs::read_to_string;
use drain::{self, ParserState, RecordsParsed, RecordParsed, RecordsParsedResult, print_log};

use log::info;
use serde_json::json;
use simple_logger::SimpleLogger;

#[test]
fn test_a() {
    print_log("apache-short.log");
    assert_eq!(1, 0);
}