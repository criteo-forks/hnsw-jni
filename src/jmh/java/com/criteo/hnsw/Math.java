package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

public class Math extends BaseBench {
    public FloatByteBuf vector1;
    public FloatByteBuf vector2;
    public MathLib math = new MathLib(defaultDimension);

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        vector1 = index.getItem(randomId());
        vector2 = index.getItem(randomId());
    }

    @Benchmark
    public float l2Norm() throws Exception {
        return math.l2NormSquared(vector1);
    }

    @Benchmark
    public float l2Dist() throws Exception {
        return math.l2DistanceSquared(vector1, vector2);
    }

    @Benchmark
    public float dotProduct() throws Exception {
        return math.innerProductDistance(vector1, vector2);
    }

    @TearDown(Level.Invocation)
    public void tearDownIter() throws Exception {
        vector1.close();
        vector2.close();
    }
}
