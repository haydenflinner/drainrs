use drainrs::print_log;

use simple_logger::SimpleLogger;

#[test]
fn test_a() {
    SimpleLogger::new().init().unwrap();
    // print_log("apache-short.log");
    print_log("Apache.log", false);
    assert_eq!(1, 1);
}
