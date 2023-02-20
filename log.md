# Log

## Downloading apache log
https://zenodo.org/record/3227177/files/Apache.tar.gz?download=1

## Benchmarking

Current results with less-copy.


10 sample size.
56482 lines many months; 5MB log size.
about 10 microseconds per line? Pretty good!
```
windows gaming pc: time:   [706.53 ms 708.81 ms 711.15 ms]
linux laptop: time:   [1.3717 s 1.5405 s 1.7036 s]
or, 1 second flat on linux. Maybe difference was that laptop was plugged in?

then.. 63ms by tweaking to match default params from drain3.ini. Note we don't have max_clusters
or an lru cache.
```


Man, rust is amazing.
`CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --bench my_benchmark -- --bench`




Looking better after tweaking to default params from drain3 ini. Now our flame graph shows plenty of time spent in extend reallocs.
looks like we could cut it in half by lifting that vector out and reusing it.
Also a noticeable portion spent in hashing, so we'll switch hashers too soon.
```
example-group/parse apache
                        time:   [64.145 ms 67.749 ms 69.332 ms]
                        change: [-93.836% -93.510% -93.173%] (p = 0.00 < 0.05)
                        Performance has improved.
```

Removed one copy seen in the trace, couldn't remove the other one due to lifetime issues.
SOmething about Iterator::collect? Found these awesome links tho:

https://stackoverflow.com/a/42639814 <-- 'most imformative rust post on SO'?
https://stackoverflow.com/a/68607168 <-- this solves my problem, if I had actually read it :-) It's great to see my suspicion of where the data needed to live, confirmed.

```
     Running benches/my_benchmark.rs (/home/hayden/code/rdrain/target/release/deps/my_benchmark-08b285c50312e958)
example-group/parse apache
                        time:   [45.788 ms 46.223 ms 46.647 ms]
                        change: [-17.199% -14.597% -11.708%] (p = 0.00 < 0.05)
                        Performance has improved.
```


Switched out the std HashMap for FxHashMap.
```
example-group/parse apache
                        time:   [41.629 ms 41.742 ms 41.925 ms]
                        change: [-44.187% -34.517% -23.028%] (p = 0.00 < 0.05)
                        Performance has improved.
```


Seeing some time spent in cloning to string to look up in map. This can fix that:
https://github.com/rust-lang/rust/issues/56167#issuecomment-468732127
ALso switch to streaming iterator will stop allocations on every line.

Before we do those hard changes, let's speed up the output since it's a big chunk remaining.

https://lovasoa.github.io/json_in_type/


```
example-group/parse apache
                        time:   [35.525 ms 37.009 ms 38.449 ms]
                        change: [-41.834% -37.323% -32.172%] (p = 0.00 < 0.05)
                        Performance has improved.
```

There must be some way to avoid calling to_string() in the common case of walking through the tree.
https://github.com/rust-lang/rust/issues/56167#issuecomment-468732127
But it's a lot of work and we're already faster than python despite not having an lru-cache.

Let's use the entry API to make sure we aren't calling to_string at every step along walking the map.
Holy cow. Will be difficult because key type is not string, it's an enum over string. No-thanks.
http://idubrov.name/rust/2018/06/01/tricking-the-hashmap.html#fnref:3

Let's go for an LRU cache instead. It looks like we spend about half of our time tokenizing and about half of our time tree searching.
In the common case of repeated lines when logging contents of a list, an LRU cache would be fine.
Though we still need to tokenize to be able to use our template.


Actually, all we did was switch to a callback rather than using an iterator (and moving a vector at each return.)
```
example-group/parse apache
                        time:   [29.704 ms 29.849 ms 30.022 ms]
                        change: [-15.626% -10.211% -5.4188%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
```
ASlways one outlier, not sure why.
Note that we went back to 36ms when capturing the flamegraph; this is the first time i've noticed a performance overhead to tracing this!
