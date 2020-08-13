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
        try(KnnResult result = index.search(queryVector, defaultNbResults)) {
            return result;
        }
    }

    @Benchmark
    public FloatByteBuf get() throws Exception {
        try(FloatByteBuf decoded = index.getItem(randomLabel)) {
            return decoded;
        }
    }

    @Benchmark
    public FloatByteBuf getDecoded() throws Exception {
        try(FloatByteBuf decoded = index.getItemDecoded(randomLabel)) {
            return decoded;
        }
    }

    @TearDown(Level.Invocation)
    public void tearDownIter() throws Exception { queryVector.close(); }
}