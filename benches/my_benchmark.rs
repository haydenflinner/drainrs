#![allow(unused_imports)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use drain::print_log;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

//fn criterion_benchmark(c: &mut Criterion) {
    //c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
//}
fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("example-group");
    group.sample_size(10);
    group.bench_function("parse apache", |b| b.iter(|| 
        print_log("Apache.log")
        ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);