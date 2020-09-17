package com.criteo.hnsw;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.Index;
import com.criteo.knn.knninterface.KnnResult;
import com.criteo.knn.knninterface.Metric;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class HelloHnswWrapped {
    public static Random r = new Random();

    public static void main(String[] args) throws Exception {
        int dimension = 100;
        int nbItems = 15000;
        int M = 16;
        int efConstruction = 200;
        int efSearch = 50;

        Map<String, String> params = new HashMap<>();
        params.put("M", Integer.toString(M));
        params.put("efConstruction", Integer.toString(efConstruction));
        params.put("efSearch", Integer.toString(efSearch));
        params.put("maxElements", Integer.toString(nbItems));
        params.put("randomSeed", "42");
        params.put("precision", "float16");

        Index index = HnswIndexWrapped.create(Metric.Euclidean, dimension, params);

        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            index.addItem(vector, i);
        }

        String path = new File("/tmp", "index.hnswWrapped").toString();

        index.writeIndex(path);
        index.readIndex(path);
        System.out.println("Loaded " + index.getNbItems() + " items");

        for (long i = 0; i < 20; i++)
            try (FloatByteBuf query = index.reconstruct(i)) {
                try (KnnResult result = index.search(query, 3)) {
                    for (int j = 0; j < result.resultCount; j++)
                        System.out.println(i + " -> " + result.resultItems[j] + " distance: " + result.resultDistances[j]);
                }
            }
        index.destroy();
    }

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = r.nextFloat();
        }
        return vector;
    }
}
