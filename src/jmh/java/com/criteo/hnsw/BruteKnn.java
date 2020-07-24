package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class BruteKnn extends BaseBench {
    public FloatByteBuf queryVector;
    public long randomLabel;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createBruteforceIndex(defaultDimension, metric, precision, defaultNbItems);
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        queryVector = index.getItem(randomId());
        randomLabel = randomId();
    }

    @Benchmark
    public KnnResult search() throws Exception {
        return index.search(queryVector, defaultNbResults);
    }

    @Benchmark
    public FloatByteBuf get() {
        return index.getItem(randomLabel);
    }

    @Benchmark
    public FloatByteBuf getDecoded() {
        return index.getItemDecoded(randomLabel);
    }
}