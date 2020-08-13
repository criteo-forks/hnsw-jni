package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class HnswKnn extends BaseBench {
    public FloatByteBuf queryVector;

    @Param({Metrics.DotProduct, Metrics.Euclidean})
    public String metric;

    @Param({Precision.Float32, Precision.Float16})
    public String precision;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
    }

    @Setup(Level.Invocation)
    public void setupIter() throws Exception {
        queryVector = index.getItemDecoded(randomId());
    }

    @Benchmark
    public KnnResult search() throws Exception {
        try(KnnResult result = index.search(queryVector, defaultNbResults)) {
            return result;
        }
    }

    @TearDown(Level.Invocation)
    public void tearDownIter() throws Exception { queryVector.close(); }
}