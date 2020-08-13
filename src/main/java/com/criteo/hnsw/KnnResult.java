package com.criteo.hnsw;

public class KnnResult implements AutoCloseable {
    public int resultCount;
    public long[] resultItems;
    public float[] resultDistances;
    public FloatByteBuf[] resultVectors;

    public KnnResult() {

    }

    @Override
    public final void close() throws Exception {
        for(FloatByteBuf vector: resultVectors) {
            vector.close();
        }
    }
}
