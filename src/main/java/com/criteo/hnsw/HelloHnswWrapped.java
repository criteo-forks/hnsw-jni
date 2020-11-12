package com.criteo.hnsw;

import com.criteo.knn.knninterface.FloatByteBuf;
import com.criteo.knn.knninterface.Index;
import com.criteo.knn.knninterface.KnnResult;
import com.criteo.knn.knninterface.Metric;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.lang.reflect.Type;
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

        Metric metric = Metric.Euclidean;

        Index index = HnswIndexWrapped.create(metric, dimension, params);

        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            index.addItem(vector, i);
        }

        String path = new File("/tmp", "index.hnswWrapped").toString();

        index.writeIndex(path);

        String paramJSON = ((HnswIndexWrapped) index).getIndexParameters();
        Gson gson = new Gson();
        Type type = new TypeToken<Map<String, String>>() {
        }.getType();
        Map<String, String> indexParams = gson.fromJson(paramJSON, type);
        index = HnswIndexWrapped.loadIndex(metric, dimension, path, indexParams);

        System.out.println("Loaded " + index.getNbItems() + " items");

        for (long i = 0; i < 20; i++)
            try (FloatByteBuf query = index.reconstruct(i)) {
                try (KnnResult result = index.search(query, 3)) {
                    for (int j = 0; j < result.resultCount; j++)
                        System.out.println(i + " -> " + result.resultItems[j] + " distance: " + result.resultDistances[j]);
                }
            }
        System.out.println("parameters:");
        System.out.println(((HnswIndexWrapped) index).getIndexParameters());
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
