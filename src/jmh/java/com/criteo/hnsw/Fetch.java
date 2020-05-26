package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class Fetch extends BaseBench  {
    public long randomLabel;
    public HnswIndex index;
    public Decoder decoder;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
        decoder = Decoder.create(index.getDimension(), precision);
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
    public FloatByteBuf getAndDecode() {
        FloatByteBuf buf = index.getItem(randomLabel);
        return decoder.decode(buf);
    }
}
