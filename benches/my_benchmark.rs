#![allow(unused_imports)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use drain::print_log;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("example-group");
    group.sample_size(10);
    group.bench_function("parse apache", |b| b.iter(||
        print_log("Apache.log", false)
        ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
