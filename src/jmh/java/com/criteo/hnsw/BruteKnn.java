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
    public void search() throws Exception {
        KnnResult result = index.search(queryVector, defaultNbResults);
        result.close();
    }

    @Benchmark
    public void get() throws Exception {
         FloatByteBuf decoded = index.getItem(randomLabel);
         decoded.close();
    }

    @Benchmark
    public void getDecoded() throws Exception {
        FloatByteBuf decoded = index.getItemDecoded(randomLabel);
        decoded.close();
    }

    @TearDown(Level.Invocation)
    public void tearDownIter() throws Exception { queryVector.close(); }
}