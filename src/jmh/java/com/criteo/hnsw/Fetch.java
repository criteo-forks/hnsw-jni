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
    public void get() throws Exception {
        FloatByteBuf item = index.getItem(randomLabel);
        item.close();
    }

    @Benchmark
    public void getDecoded() throws Exception {
        FloatByteBuf decoded = index.getItemDecoded(randomLabel);
        decoded.close();

    }
}
