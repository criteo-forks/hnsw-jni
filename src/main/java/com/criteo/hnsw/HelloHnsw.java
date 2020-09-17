package com.criteo.hnsw;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.KnnResult;

import java.io.File;
import java.util.Random;

public class HelloHnsw {
    public static Random r = new Random();

    public static void main(String[] args) throws Exception {
        int dimension = 100;
        int nbItems = 15000;
        int M = 16;
        int efConstruction = 200;

        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        index.initNewIndex(nbItems, M, efConstruction, 42);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            index.addItem(vector, i);
        }

        String path = new File("/tmp", "index.hnsw").toString();

        index.save(path);

        index.load(path);
        System.out.println("Loaded " + index.getNbItems() + " items");
        long[] ids = index.getIds();

        for (int i = 0; i < 20; i++) {
            long queryId = ids[i];
            try(FloatByteBuf query = index.getItem(queryId)){
                try(KnnResult result = index.search(query, 3)){
                    for (int j = 0; j < result.resultCount; j++) {
                        System.out.println(queryId + " -> " + result.resultItems[j] + " distance: " + result.resultDistances[j]);
                    }
                }
            }
        }
        index.unload();
    }

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = r.nextFloat();
        }
        return vector;
    }
}
