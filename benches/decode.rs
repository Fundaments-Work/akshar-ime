use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_akshara_segment(c: &mut Criterion) {
    c.bench_function("segment_namaste", |b| {
        b.iter(|| akshar_ime::core::akshara::segment(black_box("नमस्ते")))
    });
}

criterion_group!(benches, bench_akshara_segment);
criterion_main!(benches);
