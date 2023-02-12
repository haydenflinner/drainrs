use std::env;
fn main() {
    use drain::{print_log};

    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);
    print_log(&args[1], true);
}
