package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class HnswBruteKnn extends BaseBench {
    public FloatByteBuf queryVector;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
        index.enableBruteforceSearch();
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        queryVector = index.getItem(randomId());
    }

    @Benchmark
    public KnnResult search() throws Exception {
        return index.searchBruteforce(queryVector, defaultNbResults);
    }
}