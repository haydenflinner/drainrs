use std::env;
fn main() {
    use drain::{print_log};

    let args: Vec<String> = env::args().collect();
    // SimpleLogger::new().init().unwrap();
    // print_log("apache-short.log");
    assert!(args.len() == 2);
    print_log(&args[1]);
}
