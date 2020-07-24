package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class Fetch extends BaseBench  {
    public long randomLabel;
    public HnswIndex index;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        randomLabel = randomId();
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
