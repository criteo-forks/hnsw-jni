package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class HnswKnn extends BaseBench {
    public FloatByteBuf queryVector;
    public Decoder decoder;

    @Param({Metrics.DotProduct, Metrics.Euclidean})
    public String metric;

    @Param({Precision.Float32, Precision.Float16})
    public String precision;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
        decoder = Decoder.create(index.getDimension(), precision);
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        FloatByteBuf encoded = index.getItem(randomId());
        queryVector = decoder.decode(encoded);
    }

    @Benchmark
    public KnnResult search() throws Exception {
        return index.search(queryVector, defaultNbResults);
    }
}