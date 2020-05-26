package com.criteo.hnsw;

import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@BenchmarkMode({Mode.AverageTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class BaseBench {
    private static int randomSeed = 42;
    public static Random r = new Random(randomSeed);
    public static int defaultDimension = 101;

    public static int defaultM = 15;
    public static int defaultEfConstruction = 1000;
    public static int defaultEfSearch = 50;
    public static int defaultNbItems = 2500;
    public static int defaultNbResults = 20;

    @Param({Metrics.DotProduct})
    public String metric;

    @Param({Precision.Float32})
    public String precision;

    public HnswIndex index;

    public static HnswIndex createIndex(String metric, String precision) {
        return createIndex(defaultDimension, metric, precision, defaultNbItems, defaultEfConstruction, defaultM, defaultEfSearch);
    }

    public static HnswIndex createIndex(int dimension, String metric, String precision, int nbItems, int efConstruction, int m, int efSearch) {
        HnswIndex index = HnswIndex.create(metric, dimension, precision, false);
        index.initNewIndex(nbItems, m, efConstruction, randomSeed);
        populateIndex(index, nbItems, dimension);
        index.setEf(efSearch);
        return index;
    }

    public static HnswIndex createBruteforceIndex(int dimension, String metric, String precision, int nbItems) {
        HnswIndex index = HnswIndex.create(metric, dimension, precision, true);
        index.initBruteforce(nbItems);
        populateIndex(index, nbItems, dimension);
        return index;
    }

    public static void populateIndex(HnswIndex index, int nbItems, int dimension) {
        for(long id = 0; id < nbItems; id++) {
            index.addItem(createVectorArray(dimension), id);
        }
    }

    public static float[] createVectorArray(int dimension) {
        float[] embedding = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            embedding[i] = r.nextFloat();
        }
        return embedding;
    }

    public int randomId() {
        return randomId(defaultNbItems);
    }

    public int randomId(int maxId) {
        return BaseBench.r.nextInt(maxId);
    }

    @TearDown(Level.Trial)
    public void globalTearDown() {
        if(index != null) {
            index.unload();
        }
    }

}
