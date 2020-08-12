package com.criteo.hnsw;

public class KnnResult {
    public int resultCount;
    public long[] resultItems;
    public float[] resultDistances;
    public FloatByteBuf[] resultVectors;

    public KnnResult() {

    }

    public final void close() throws Exception {
        for(FloatByteBuf vector: resultVectors) {
            vector.close();
        }
    }
}
