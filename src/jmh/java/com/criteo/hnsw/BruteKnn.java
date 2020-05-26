package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class BruteKnn extends BaseBench {
    public FloatByteBuf queryVector;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createBruteforceIndex(defaultDimension, metric, precision, defaultNbItems);
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        queryVector = index.getItem(randomId());
    }

    @Benchmark
    public KnnResult search() throws Exception {
        return index.search(queryVector, defaultNbResults);
    }
}