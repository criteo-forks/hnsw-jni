package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.KnnResult;

public class HnswBruteKnn extends BaseBench {
    public FloatByteBuf queryVector;

    @Setup(Level.Trial)
    public void globalSetup() {
        index = createIndex(metric, precision);
        index.enableBruteforceSearch();
    }

    @Setup(Level.Invocation)
    public void setupIter() {
        queryVector = index.getItem(randomId());
    }

    @Benchmark
    public KnnResult search() throws Exception {
        try(KnnResult result = index.searchBruteforce(queryVector, defaultNbResults)) {
            return result;
        }
    }

    @TearDown(Level.Invocation)
    public void tearDownIter() throws Exception { queryVector.close(); }
}